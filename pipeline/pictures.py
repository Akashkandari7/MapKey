"""Picture cards: a photo of the place pops up over the map when it is named ("Hong Kong" -> its skyline).

    python -m pipeline.pictures --episode work/<ep>          # (re)place picture cards in scenes.json

Photos come only from Wikimedia Commons, only under licences that allow reuse in a commercial reel with credit
(public domain, CC0, CC BY, CC BY-SA), and never show people: any photo in which a face is detected is skipped
(user rule: no real people's photos). Each card carries its own credit line (author + licence), as CC BY requires.
Downloads are cached in data/pictures/ (not in git).

Called by the plan stage after scenes.json is written: it needs every text box on screen (the same maths as the
review) to put each card where it covers no label, number or the place itself.
"""
import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

from pipeline.common import ROOT, read_json, require_env, sha, write_json

API = "https://commons.wikimedia.org/w/api.php"
CACHE = ROOT / "data" / "pictures"
UA = {"User-Agent": "MapaniReels/0.1 (map explainer reels; contact via github.com/Akashkandari7/MapKey)"}
OK_LICENCES = re.compile(r"^(cc0|public domain|pd\b.*|cc[ -]by(-sa)?[ -]\d(\.\d)?.*)$", re.I)
NOT_PHOTO = re.compile(r"\b(map|maps|flag|locator|location|logo|emblem|seal|coat of arms|diagram|chart|plan|svg|"
                       r"portrait|people|crowd|selfie|protest|parade|festival|wedding|museum|exhibit|exhibition|"
                       r"pottery|ornament|artifact|artefact|statue|sculpture|coin|stamp|painting|manuscript)\b", re.I)
# India rule for photos: a photo of a place in India must never be shown as a foreign place (a Ladakh peak called "Tibet")
INDIA_WORDS = re.compile(r"\b(india|indian|ladakh|zanskar|leh|kashmir|jammu|arunachal|tawang|sikkim|himachal|"
                         r"uttarakhand|aksai|gilgit|baltistan|siachen)\b", re.I)


QUERIES_PROMPT = """For each place below (from a short map-explainer video), give the best Wikimedia Commons search query
for ONE striking, well-known photo that instantly says "this place" to a viewer: its most famous landmark or landscape
(e.g. Beijing -> "Forbidden City aerial", Guangxi -> "Li River Guilin karst", Tibet -> "Potala Palace"). Scenery or
architecture only, never people, never a map, flag or satellite image. 2-5 words, in English.
Places (id: name, kind, country): """
QUERIES_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "query": {"type": "string"},
                                     "must": {"type": "string", "description": "one word the photo's title must contain"}},
    "required": ["id", "query", "must"]}}}, "required": ["queries"]}


def landmark_queries(places, cache_dir):
    """{entity id: (query, must-word)} from Gemini, cached per episode (one call for all places)."""
    from google import genai
    from google.genai import types
    from pipeline.entities import DEFAULT_MODEL
    listing = "; ".join(f"{e['id']}: {e.get('name_en') or e.get('display')}, {e.get('kind')}, {e.get('country')}" for e in places)
    f = Path(cache_dir) / f"picture_queries_{sha(listing)}.json"
    if not f.exists():
        client = genai.Client(api_key=require_env("GEMINI_API_KEY", "needed to choose picture searches"))
        resp = client.models.generate_content(model=DEFAULT_MODEL, contents=[QUERIES_PROMPT + listing], config=types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json", response_json_schema=QUERIES_SCHEMA))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(resp.text, encoding="utf-8")
    return {q["id"]: (q["query"], q["must"]) for q in json.loads(f.read_text(encoding="utf-8"))["queries"]}


def query_for(e):
    """What to search for: the entity's own hint, else a landmark-style photo of the place."""
    if e.get("picture_query"):
        return e["picture_query"]
    name = e.get("name_en") or e.get("display")
    kind = (e.get("kind") or "").lower()
    if kind in ("city", "town", "district", "neighbourhood"):
        return f"{name} skyline"
    if kind in ("region", "state", "province", "island", "country"):
        return f"{name} landscape"
    return f"{name} landmark"


def _strip(h):
    return html.unescape(re.sub(r"<[^>]+>", "", h or "")).strip()


def has_people(path):
    """Faces (front or side) or standing people anywhere in the photo: then it isn't used."""
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        return True
    k = 1000 / max(im.shape[:2])
    im = cv2.resize(im, None, fx=k, fy=k)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    for xml in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        det = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
        if len(det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(22, 22))):
            return True
    hog = cv2.HOGDescriptor()  # pedestrians: tourists in a street photo have no frontal face to find
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    _, weights = hog.detectMultiScale(im, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return bool(len(weights)) and float(np.max(weights)) > 0.5


def _get(url, **kw):
    """Polite GET: Wikimedia asks for a slow request rate; on 429 wait as told and retry."""
    import time
    import requests
    for attempt in range(5):
        time.sleep(1.5)
        r = requests.get(url, headers=UA, timeout=60, **kw)
        if r.status_code != 429:
            return r
        time.sleep(float(r.headers.get("Retry-After") or 5 * (attempt + 1)))
    r.raise_for_status()
    return r


def fetch(query, n_candidates=20, must=None, indian=False):
    """Best usable photo for a query: {file, title, credit, licence, page} or None. Cached by query. `must`: a word the
    photo's title or description has to contain (search matches loosely: 'Beijing' found a balloon festival)."""
    import requests
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_f = CACHE / f"q_{sha(query, must, indian, 2)}.json"
    if meta_f.exists():
        m = json.loads(meta_f.read_text(encoding="utf-8"))
        if not m or (CACHE / m["file"]).exists():
            return m or None
    pages = []
    # Commons' reviewed "Quality images" first (sharp, well exposed), then any photo
    for extra in (" incategory:Quality_images", ""):
        r = _get(API, params={
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
            "gsrsearch": f"{query} filetype:bitmap filew:>1400{extra}", "gsrlimit": n_candidates,
            "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 1280})  # a standard thumb size (odd sizes are rendered on demand and throttled)
        r.raise_for_status()
        pages += sorted(r.json().get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    found, seen = None, set()
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata", {})
        lic = _strip(md.get("LicenseShortName", {}).get("value"))
        title = p["title"]
        if title in seen:
            continue
        desc = _strip(md.get("ImageDescription", {}).get("value"))[:400]
        text = (title + " " + desc).lower()
        if must and must.lower() not in text:
            continue
        # the right place, not a namesake: at least two of the query's words ("Potala" alone found a Zanskar peak)
        words = set(re.findall(r"[a-z]{4,}", query.lower()))
        if len(words) >= 2 and sum(w in text for w in words) < 2:
            continue
        if not indian and INDIA_WORDS.search(text):
            continue
        seen.add(title)
        w, h = ii.get("width", 0), ii.get("height", 0)
        if not OK_LICENCES.match(lic) or ii.get("mime") not in ("image/jpeg", "image/png") or not h \
                or not 1.25 <= w / h <= 2.6 or NOT_PHOTO.search(title) \
                or NOT_PHOTO.search(_strip(md.get("ImageDescription", {}).get("value"))[:300]):
            continue
        f = CACHE / f"p_{sha(title)}.jpg"
        if not f.exists():
            try:
                img = _get(ii["thumburl"])
            except Exception as ex:  # noqa: BLE001  (throttled / gone: try the next photo)
                print(f"   skip {title}: {ex}")
                continue
            if img.status_code != 200:
                continue
            f.write_bytes(img.content)
        if has_people(f):  # never real people's photos
            continue
        artist = _strip(md.get("Artist", {}).get("value")) or "Wikimedia Commons"
        artist = re.sub(r"\s+", " ", artist)[:40]
        found = {"file": f.name, "title": title, "licence": lic, "page": ii.get("descriptionurl"),
                 "credit": f"{artist} · {lic}"}
        break
    meta_f.write_text(json.dumps(found or {}, ensure_ascii=False, indent=1), encoding="utf-8")
    return found


def add(ep, tokens=None):
    """Picture cards for the places named while the map is on: each card pops up on its word and stays until the next
    card (a quick list of names becomes a quick photo montage) or a few seconds, in the top or bottom band of the
    frame, whichever covers no text, label or the place itself at that moment."""
    from pipeline.review import Reviewer
    ep = Path(ep)
    T = tokens or read_json(ROOT / "style/tokens.json")
    PC = T["looks"][read_json(ep / "scenes.json")["look"]]["picture"]
    rv = Reviewer(ep)
    sc = rv.sc
    shot = next((s for s in sc["shots"] if s["kind"] == "map"), None)
    if not shot or not PC.get("enabled", True):
        return 0
    shot["layers"] = [l for l in shot["layers"] if l["type"] != "picture"]
    main = rv.ents_doc.get("main_place")
    places = [e for e in rv.ents.values() if e["type"] == "place" and e["id"] != main
              and shot["start"] <= e["start"] < shot["end"] - 0.8]
    places.sort(key=lambda e: e["start"])
    W, H = rv.W, rv.H
    sl, st, sr, sb = rv.safe

    def boxes_at(t, own_ent):
        out = []
        for l in shot["layers"]:
            if l["type"] not in ("title", "stat", "pin"):
                continue
            t0 = l.get("in", l["lines"][0]["in"] if l.get("lines") else 0)
            if not (t0 - 0.1 <= t <= (l.get("out") or shot["end"]) + 0.3):
                continue
            b = rv.text_box(shot, l, t)
            if b:
                out.append(b)
            if l["type"] == "pin" and l.get("ent") == own_ent:  # the place itself (its dot) stays visible
                X, Y = sc["geometry"][l["geom"]]["coords"]
                x, y = rv.to_px(rv.cam(shot, t), X, Y)
                out.append((x - 90, y - 90, x + 90, y + 90))
        return out

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    try:
        qs = landmark_queries(places, ep / "cache")
    except Exception as ex:  # noqa: BLE001  (no key / offline: plain name searches still work)
        print(f"[pictures] landmark queries unavailable ({ex}); searching by name")
        qs = {}
    cards, prev = [], None
    for i, e in enumerate(places):
        q, must = qs.get(e["id"], (query_for(e), None))
        indian = (e.get("country") or "").upper() == "IN"
        pic = fetch(q, must=must, indian=indian) or             fetch(query_for(e), must=(e.get("name_en") or "").split()[0] or None, indian=indian)
        if not pic:
            print(f"[pictures] no usable photo for {e.get('display')!r}")
            continue
        t_in = round(e["start"] + PC["in_after_word"], 3)
        nxt = next((p["start"] for p in places[i + 1:]), shot["end"])
        t_out = round(min(nxt + PC["in_after_word"], t_in + PC["max_s"], shot["end"] - 0.1), 3)
        best = None
        for w, h in PC["sizes"]:
            for band in ("top", "bottom"):
                y = st + 30 + h / 2 if band == "top" else sb - 30 - h / 2
                box = (W / 2 - w / 2, y - h / 2, W / 2 + w / 2, y + h / 2)
                cover = max(sum(overlap(box, b) for b in boxes_at(t, e["id"])) for t in np.arange(t_in, t_out + 0.01, 0.25))
                cover /= w * h
                keep_band = prev and prev["band"] == band and abs(prev["out"] - t_in) < 0.3
                score = cover - (0.02 if keep_band else 0)
                if best is None or score < best[0]:
                    best = (score, cover, w, h, y, band)
            if best and best[1] <= PC["max_cover"]:
                break
        score, cover, w, h, y, band = best
        if cover > PC["max_cover"]:
            print(f"[pictures] {e.get('display')!r}: no free spot (covers {cover:.0%}), skipped")
            continue
        card = {"type": "picture", "ent": e["id"], "src": f"data/pictures/{pic['file']}", "caption": e.get("display")
                or e.get("name_en"), "credit": pic["credit"], "page": pic["page"], "x": W / 2, "y": round(y, 1),
                "w": w, "h": h, "in": t_in, "out": t_out, "band": band}
        cards.append(card)
        prev = card
    shot["layers"] += cards
    if cards:
        sc["credits"] = sorted(set(sc.get("credits", [])) | {"Photos: Wikimedia Commons (credits on each photo)"})
    write_json(ep / "scenes.json", sc)
    print(f"[pictures] {len(cards)} picture cards")
    return len(cards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--query", help="only search: print the photo a query would use")
    a = ap.parse_args()
    if a.query:
        print(fetch(a.query))
        return
    add(a.episode)


if __name__ == "__main__":
    main()
