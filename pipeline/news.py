"""News clippings for story mode: a real article's headline + first paragraphs as a picture, and where the spoken
phrases are on it (for the marker sweep).

    python -m pipeline.news --url https://... --out work/<ep>/news/n1.png --phrase "100 per cent" --phrase "RBI"
    python -m pipeline.news --find "Delhi groundwater over-exploited RBI report" --out work/<ep>/news/n1.png

Articles come from links the user gives (inputs/<episode>/news.txt or the storyboard), or, when asked, from a search:
Gemini with Google Search grounding proposes article URLs from known news sites, and the first one that loads with a
headline is used. Every clipping used is listed in the review report with its URL, for the user to check.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pipeline.common import ROOT, require_env, sha

NODE_SCRIPT = ROOT / "pipeline" / "render" / "news_shot.mjs"


def boxes_from_diff(plain, marked):
    """Marker boxes (x, y, w, h) per text line: pixels painted magenta in `marked` but not in `plain`, grouped into
    runs of rows (one run = one line of the phrase)."""
    from PIL import Image
    a = np.asarray(Image.open(plain).convert("RGB")).astype(np.int16)
    b = np.asarray(Image.open(marked).convert("RGB")).astype(np.int16)
    # changed AND magenta: anything else that changed between the two shots (a carousel, a clock) is not the marker
    m = (np.abs(a - b).sum(2) > 60) & (b[..., 0] > 170) & (b[..., 2] > 170) & (b[..., 1] < 110)
    rows = [y for y in range(m.shape[0]) if m[y].sum() > 6]
    if not rows:
        return []
    # a new box where the rows stop touching, or where the painted span jumps (a headline's lines touch each other:
    # line 2 starts at the left margin while line 1 ended mid-line)
    span = {y: (np.nonzero(m[y])[0].min(), np.nonzero(m[y])[0].max()) for y in rows}
    groups = [[rows[0]]]
    for prev, y in zip(rows, rows[1:]):
        (a0, a1), (b0, b1) = span[prev], span[y]
        if y > prev + 3 or abs(a0 - b0) > 24 or abs(a1 - b1) > 24:
            groups.append([])
        groups[-1].append(y)
    out = []
    for g in groups:
        x0 = int(np.median([span[y][0] for y in g]))
        x1 = int(np.median([span[y][1] for y in g]))
        box = [x0, g[0], x1 - x0 + 1, g[-1] - g[0] + 1]
        p = out[-1] if out else None
        if p and box[1] <= p[1] + p[3] + 3 and abs(p[0] - box[0]) <= 24 and abs(p[0] + p[2] - box[0] - box[2]) <= 24:
            p[3] = box[1] + box[3] - p[1]  # descenders split one line in two: join them back
            continue
        out.append(box)
    return [o for o in out if o[3] >= 8]


def capture(url, out, phrases=(), height=640, images=False):
    """Screenshot the article at url into out (PNG). Returns {"ok", "title", "size", "url", "phrases": {p: boxes}}."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["node", str(NODE_SCRIPT), f"--url={url}", f"--out={out.resolve()}", f"--height={height}",
           f"--phrases={json.dumps(list(phrases), ensure_ascii=False)}"] + (["--images"] if images else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=150)
    except subprocess.TimeoutExpired:  # a page that never settles (endless ads / scripts): next candidate
        return {"ok": False, "why": "page took over 150 s", "url": url}
    line = next((ln for ln in r.stdout.splitlines()[::-1] if ln.startswith("{")), None)
    if not line:
        return {"ok": False, "why": (r.stderr or r.stdout)[-400:], "url": url}
    res = json.loads(line)
    res["url"] = url
    if not res.get("ok"):
        return res
    res["phrases"] = {}
    for p, shot in zip(phrases, res.pop("highlight_shots", [])):
        res["phrases"][p] = boxes_from_diff(out, shot) if shot else []
        if shot:
            Path(shot).unlink(missing_ok=True)
    logo = res.pop("logo", None)
    if logo and Path(logo).exists():
        k = add_masthead(out, logo)
        for p in res["phrases"]:
            res["phrases"][p] = [[x, y + k, w, h] for x, y, w, h in res["phrases"][p]]
        res["size"][1] += k
        res["masthead"] = True
        Path(logo).unlink(missing_ok=True)
    return res


def add_masthead(shot, logo):
    """The paper's own logo centred above the article, like a printed front page (q2 'Hindustan Times', q4 'The
    Tribune'). A white logo (made for a dark header bar) gets a dark strip. Returns the strip's height."""
    from PIL import Image, ImageDraw
    im = Image.open(shot).convert("RGB")
    lg = Image.open(logo).convert("RGBA")
    a = np.asarray(lg).astype(np.float32)
    opaque = a[..., 3] > 128
    lum = (a[..., :3][opaque] @ [0.299, 0.587, 0.114]).mean() if opaque.any() else 0
    h = 92
    lg = lg.resize((max(1, round(lg.width * h / lg.height)), h), Image.LANCZOS)
    if lg.width > im.width * 0.72:
        lg = lg.resize((round(im.width * 0.72), max(1, round(lg.height * im.width * 0.72 / lg.width))), Image.LANCZOS)
    strip = lg.height + 84
    dark = lum > 200
    out = Image.new("RGB", (im.width, im.height + strip), (18, 18, 20) if dark else (255, 255, 255))
    out.paste(lg, ((im.width - lg.width) // 2, 40), lg)
    ImageDraw.Draw(out).line([(40, strip - 10), (im.width - 40, strip - 10)], fill=(90, 90, 90) if dark else (200, 200, 200), width=3)
    out.paste(im, (0, strip))
    out.save(shot)
    return strip


def find_articles(query, n=5, model="gemini-3.8-flash"):
    """Article URLs about `query` from real news sites, via Gemini + Google Search grounding (the grounding chunks are
    the real pages it read; we never trust a URL the model only wrote)."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=require_env("GEMINI_API_KEY", "put GEMINI_API_KEY=... in .env"))
    resp = client.models.generate_content(
        model=model,
        contents=f"Find {n} recent news articles (Indian or international newspapers / news sites, not Wikipedia, "
                 f"not YouTube, not social media) about: {query}. List their headlines.",
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0))
    urls = []
    for cand in resp.candidates or []:
        gm = getattr(cand, "grounding_metadata", None)
        for ch in (gm.grounding_chunks if gm and gm.grounding_chunks else []):
            if ch.web and ch.web.uri and ch.web.uri not in urls:
                urls.append(ch.web.uri)
    return urls


def resolve_redirect(url):
    """Grounding links are Google redirect URLs: follow them to the article's own address."""
    import requests
    try:
        r = requests.get(url, allow_redirects=True, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        return r.url
    except Exception:  # noqa: BLE001
        return url


def supports(claim, title, text, model="gemini-3.8-flash"):
    """Does the article say what the narration says at this point? A found article that contradicts the voice
    ('groundwater improves' under 'Delhi over-exploits its groundwater') must never be shown."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=require_env("GEMINI_API_KEY", "put GEMINI_API_KEY=... in .env"))
    r = client.models.generate_content(
        model=model,
        contents=["A reel's narration says (Hindi/Hinglish, may be loose):\n" + claim +
                  "\n\nA news article's headline and opening:\n" + title + "\n" + (text or "")[:1500] +
                  "\n\nWould an editor show this article on screen at this moment as evidence for what is said? It must be "
                  "about the same event/report/decision, and its HEADLINE (the viewer reads it) must point the same way as "
                  "the narration: if the voice says a problem is bad/getting worse and the headline says it improved (or "
                  "the reverse), or the article is about a different aspect, answer ok=false. Answer JSON."],
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_json_schema={
            "type": "object", "properties": {"ok": {"type": "boolean"}, "why": {"type": "string"}}, "required": ["ok", "why"]}))
    v = json.loads(r.text)
    return bool(v.get("ok")), v.get("why", "")


def clipping(out, phrases=(), url=None, query=None, cache=True, claim=None, more_queries=None, images=False):
    """A clipping from a given url, or the first search hit for query that loads with a headline (and, when claim is
    given, says the same thing as the narration)."""
    out = Path(out)
    meta = out.with_suffix(".json")
    key = sha(url, query, list(phrases), claim, images)
    if cache and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("key") == key and out.exists():
            return m
    tried = []
    if url:
        urls = [url]
    else:  # search answers vary run to run: a second, wider search when the first gives nothing that fits
        urls = []
        for q in [query] + list(more_queries or []):
            urls += [u for u in (resolve_redirect(x) for x in find_articles(q)) if u not in urls]
    for u in urls:
        if any(s in u for s in ("youtube.", "wikipedia.", "facebook.", "instagram.", "x.com", "twitter.")):
            continue
        res = capture(u, out, phrases, images=images)
        if res.get("ok") and claim and not url:  # a link the user gave is trusted; a search hit is checked
            fits, why = supports(claim, res.get("title", ""), res.get("text", ""))
            if not fits:
                res = {"ok": False, "why": "does not match the narration: " + why}
        tried.append({"url": u, "ok": res.get("ok"), "why": res.get("why")})
        if res.get("ok"):
            res.update({"key": key, "query": query, "tried": tried})
            meta.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            return res
    return {"ok": False, "tried": tried, "query": query, "url": url}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--find")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phrase", action="append", default=[])
    a = ap.parse_args()
    if not a.url and not a.find:
        ap.error("give --url or --find")
    res = clipping(a.out, a.phrase, a.url, a.find, cache=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
