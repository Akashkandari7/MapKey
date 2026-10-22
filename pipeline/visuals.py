"""Story mode, stage 3c: storyboard.json -> visuals.json + the pictures (AI photos, news clippings, end-card assets).

    python -m pipeline.visuals --episode work/<ep> [--find-news] [--workers 3]

- image beats: Gemini image model, 9:16 at 2K (1536x2752). A recurring character's first picture is passed as a
  reference to every later one, so the same person keeps the same face and clothes. Cached by prompt.
- news beats: the user's links (inputs/<episode>/news.txt, one URL per line, used in order), or with --find-news a
  search (pipeline/news.py). A news beat without an article becomes a photo of what is said (never a made-up
  newspaper).
- cta / follow: the CTA backdrop is the previous photo (blurred later); the end card uses inputs/brand/avatar.* (and
  inputs/brand/profile.* as its dimmed backdrop if present).

visuals.json: {"b03": {"kind": "image", "src": "work/ep/visuals/b03.png", "prompt": ...}, "b04": {"kind": "news",
"src": ..., "url": ..., "title": ..., "boxes": {"phrase": [[x, y, w, h], ...]}}, ...}
"""
import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

from pipeline.common import ROOT, read_json, require_env, sha, write_json

IMAGE_MODEL = "gemini-3.1-flash-image"
STYLE = ("Photorealistic, cinematic colour grading, dramatic natural light, rich detail, 35mm lens look, vertical 9:16 "
         "composition with the subject in the middle third and calm space at the top and bottom for captions. "
         "No text, no letters, no numbers, no logos, no watermark, no borders.")
STYLE_ILLUSTRATION = ("High-end 3D render, soft studio light, rich detail, vertical 9:16 composition, subject centred, "
                      "calm space at top and bottom. No text, no letters, no numbers, no logos, no watermark.")


def full_prompt(b, characters):
    p = b.get("image_prompt") or b.get("note", "")
    illus = any(k in p.lower() for k in ("isometric", "3d render", "illustration", "cross-section", "diagram"))
    ch = characters.get(b.get("character") or "", "")
    if ch:
        p += f" The person is: {ch}."
    return f"{p}\n\n{STYLE_ILLUSTRATION if illus else STYLE}"


def gen_image(prompt, out, refs=(), model=IMAGE_MODEL, tries=3, aspect="9:16"):
    from google import genai
    from google.genai import types
    from PIL import Image
    client = genai.Client(api_key=require_env("GEMINI_API_KEY", "put GEMINI_API_KEY=... in .env"))
    contents = [Image.open(r) for r in refs] + [
        ("Same person as in the reference picture (same face, hair, clothes). " if refs else "") + prompt]
    last = None
    for k in range(tries):
        try:
            r = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(
                response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio=aspect, image_size="2K")))
            for part in (r.candidates[0].content.parts if r.candidates and r.candidates[0].content else []):
                if part.inline_data and part.inline_data.data:
                    Path(out).write_bytes(part.inline_data.data)
                    return True
            last = f"no image in the answer ({getattr(r.candidates[0], 'finish_reason', '?') if r.candidates else 'blocked'})"
        except Exception as e:  # noqa: BLE001
            last = str(e)[:300]
        time.sleep(4 * (k + 1))
    print(f"[visuals] image failed: {last}", file=sys.stderr)
    return False


def news_links(ep):
    f = ROOT / "inputs" / ep.name / "news.txt"
    if not f.exists():
        return []
    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip().startswith("http")]


def find_brand(stem, ep=None):
    """inputs/<episode>/<stem>.* (this reel's creator), else inputs/brand/<stem>.*"""
    for d in ([ROOT / "inputs" / ep.name] if ep else []) + [ROOT / "inputs" / "brand"]:
        pics = sorted(d.glob(f"{stem}.*")) if d.exists() else []
        hit = next((p for p in pics if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")), None)
        if hit:
            return hit
    return None


def avatar_from_facecam(ep, out):
    """No profile photo given: the creator's face from their own face-cam clip, square, for the Follow card."""
    import cv2
    d = ROOT / "inputs" / ep.name
    clips = sorted(d.glob("*cam*.*")) if d.exists() else []
    clip = next((p for p in clips if p.suffix.lower() in (".mp4", ".mov", ".mkv")), None)
    if not clip:
        return None
    fw = read_json(ep / "facecam_windows.json") if (ep / "facecam_windows.json").exists() else {}
    wins, off = fw.get("windows") or [[1.0, 3.0]], fw.get("offset", 0)
    cap = cv2.VideoCapture(str(clip))
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    best = None
    for a, b in wins:
        for t in (a + (b - a) * k / 4 for k in range(1, 4)):
            cap.set(cv2.CAP_PROP_POS_MSEC, (t - off) * 1000)
            ok, fr = cap.read()
            if not ok:
                continue
            for (x, y, w, h) in cas.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 1.1, 6, minSize=(90, 90)):
                if best is None or w > best[0][2]:
                    best = ((x, y, w, h), fr)
    if not best:
        return None
    (x, y, w, h), fr = best
    side = int(w * 2.3)
    cx, cy = x + w // 2, y + int(h * 0.62)
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    cv2.imwrite(str(out), fr[y0:y0 + side, x0:x0 + side])
    return out


def build(ep, find_news=False, workers=3):
    from pipeline.news import clipping
    ep = Path(ep).resolve()
    sb = read_json(ep / "storyboard.json")
    words = read_json(ep / "transcript.json")["words"]
    out_dir = ep / "visuals"
    out_dir.mkdir(exist_ok=True)
    prev = read_json(ep / "visuals.json") if (ep / "visuals.json").exists() else {}
    res = {}
    links = news_links(ep)
    beats = sb["beats"]
    # news first: a news beat without an article turns into an image beat
    for b in beats:
        if b["visual"] != "news":
            continue
        n = b.get("news") or {}
        url = links.pop(0) if links else None
        if url or find_news:
            out = out_dir / f"{b['id']}_news.png"
            # the beat often stops mid-sentence ('...report says Delhi its groundwater' | 'exploits 100 %'): the check
            # reads the whole thought around it
            claim = " ".join(w["w"] for w in words[max(0, b["w0"] - 8):b["w1"] + 22]) + f" (on screen: {b.get('note', '')})"
            # cards keep the article's own photo: real people shown for real (we never generate them), and a page
            # without its photo left empty gallery boxes (AP)
            c = clipping(out, n.get("phrases", []), url=url, query=None if url else n.get("query"), claim=claim,
                         more_queries=[f"{n.get('query')} news report", b.get("note", "")], images=True)
            if c.get("ok"):
                res[b["id"]] = {"kind": "news", "src": str(out.relative_to(ROOT)).replace("\\", "/"), "url": c["url"],
                                "title": c.get("title"), "boxes": c.get("phrases", {}), "size": c.get("size")}
                print(f"[visuals] {b['id']} news: {c.get('title')!r} ({c['url']})")
                # montage: more articles on the same point (search only), each checked against the voice too
                extra, seen = [], {c["url"]}
                for k, q in enumerate((n.get("more") or [])[:2] if (find_news and not url) else []):
                    o2 = out_dir / f"{b['id']}_news{k + 2}.png"
                    c2 = clipping(o2, n.get("phrases", []), query=q, claim=claim, more_queries=[f"{q} report"], images=True)
                    if c2.get("ok") and c2["url"] not in seen:
                        seen.add(c2["url"])
                        extra.append({"src": str(o2.relative_to(ROOT)).replace("\\", "/"), "url": c2["url"],
                                      "title": c2.get("title"), "boxes": c2.get("phrases", {}), "size": c2.get("size")})
                        print(f"[visuals] {b['id']} montage +: {c2.get('title')!r} ({c2['url']})")
                if extra:
                    res[b["id"]]["extra"] = extra
                continue
            print(f"[visuals] {b['id']} news: no article loaded ({n.get('query')}) -> photo", file=sys.stderr)
            for t in c.get("tried", []):
                print(f"           tried {t['url'][:90]}: {(t.get('why') or '')[:140]}", file=sys.stderr)
        b["visual"] = "image"
        b.setdefault("image_prompt", b.get("note", ""))
        b["_was_news"] = True
    # images: characters' first pictures go first (they are the references for the later ones)
    chars = sb.get("characters", {})
    jobs = [b for b in beats if b["visual"] == "image"]
    first_of = {}
    for b in jobs:
        if b.get("character") and b["character"] not in first_of:
            first_of[b["character"]] = b["id"]
    order = sorted(jobs, key=lambda b: (b["id"] not in first_of.values(), b["start"]))

    # map beats with a picture inside the place's outline (style photo): square pictures, made with the rest
    photo_maps = [b for b in beats if b["visual"] == "map" and (b.get("map") or {}).get("style") == "photo"
                  and (b["map"].get("photo_prompt") or "").strip()]
    order += photo_maps

    def run(b):
        if b["visual"] == "map":
            prompt = (b["map"]["photo_prompt"] + " The main object big and centred, dramatic light, rich detail, "
                      "photorealistic. No text, no letters, no logos, no watermark.")
            key = sha(prompt, IMAGE_MODEL, "1:1")
            out = out_dir / f"{b['id']}_map_{key[:8]}.png"
            if not out.exists() and not gen_image(prompt, out, aspect="1:1"):
                return b["id"], None
            return b["id"], {"kind": "map_photo", "src": str(out.relative_to(ROOT)).replace("\\", "/"), "prompt": prompt,
                             "key": key}
        prompt = full_prompt(b, chars)
        ref_id = first_of.get(b.get("character") or "")
        refs = [ROOT / res[ref_id]["src"]] if ref_id and ref_id != b["id"] and ref_id in res else []
        key = sha(prompt, [str(r) for r in refs], IMAGE_MODEL)
        out = out_dir / f"{b['id']}_{key[:8]}.png"
        old = prev.get(b["id"], {})
        if out.exists() or (old.get("key") == key and (ROOT / old.get("src", "_")).exists()):
            out = out if out.exists() else ROOT / old["src"]
        elif not gen_image(prompt, out, refs):
            return b["id"], None
        return b["id"], {"kind": "image", "src": str(out.relative_to(ROOT)).replace("\\", "/"), "prompt": prompt,
                         "key": key, "model": IMAGE_MODEL, "ref": ref_id if refs else None}
    firsts = [b for b in order if b["id"] in first_of.values()]
    rest = [b for b in order if b["id"] not in first_of.values()]
    for group in (firsts, rest):
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for bid, v in ex.map(run, group):
                if v:
                    res[bid] = v
                    print(f"[visuals] {bid} image ok")
                else:
                    print(f"[visuals] {bid} image FAILED (placeholder will show)", file=sys.stderr)
    # CTA backdrop: the last photo before it; end card: the brand avatar / profile screenshot
    last_img = None
    for b in beats:
        if b["id"] in res and res[b["id"]]["kind"] == "image":
            last_img = res[b["id"]]["src"]
        if b["visual"] == "cta":
            res[b["id"]] = {"kind": "cta", "bg_src": last_img}
        if b["visual"] == "follow":
            av, prof = find_brand("avatar", ep), find_brand("profile", ep)
            own = (ROOT / "inputs" / ep.name / "brand.json").exists()
            if own and av and av.parent.name == "brand":
                av = None  # a reel for another creator must never show the default creator's photo
            if own and prof and prof.parent.name == "brand":
                prof = None
            if not av:
                av = avatar_from_facecam(ep, out_dir / "avatar_from_facecam.jpg")
            res[b["id"]] = {"kind": "follow", "avatar_src": str(av.relative_to(ROOT)).replace("\\", "/") if av else None,
                            # no profile screenshot: the last photo, blurred and dimmed, sits behind the card
                            "bg_src": str(prof.relative_to(ROOT)).replace("\\", "/") if prof else last_img,
                            "bg_is_profile": bool(prof)}
            if not av:
                print("[visuals] no inputs/brand/avatar.jpg: the Follow card shows an empty circle", file=sys.stderr)
    write_json(ep / "storyboard.json", sb)  # news beats that became photos
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--find-news", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    res = build(a.episode, a.find_news, a.workers)
    write_json(Path(a.episode) / "visuals.json", res)
    kinds = {}
    for v in res.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    print(f"[visuals] {kinds} -> {Path(a.episode) / 'visuals.json'}")


if __name__ == "__main__":
    main()
