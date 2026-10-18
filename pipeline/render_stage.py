"""Stage 5: scenes.json -> video_raw.mp4 (silent picture + narration for preview), via headless Chrome.

    python -m pipeline.render_stage --episode work/<ep> [--scenes work/<ep>/scenes.json] [--workers 3] [--sheet 0,2,4]

Prepares work/<ep>/render/:
    studio.html            loads the shared engine (pipeline/render/kit.js + mapani.js), fonts, tokens, scenes
    data.js                TOKENS + SCENES (with the imagery levels this episode needs)
    scenes/<shot id>.js    one file per shot: registers that shot with the engine
    img/*.jpg              Sentinel-2 basemap levels, colour-graded for the look
then renders frames in parallel (resumable) and encodes them.
"""
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from pipeline.common import ROOT, read_json, sha, write_json
from pipeline.geo.proj import from_3857

ENGINE = ROOT / "pipeline" / "render"
LEVELS = [160, 80, 40, 20, 10]   # metres per pixel of the available basemap levels (Sentinel-2 native = 10)
MAX_PX = 40e6


def views(shot, scenes, step=0.25):
    """(xmin, ymin, xmax, ymax, mpp) of the camera view every `step` seconds, same camera maths as mapani.js."""
    from pipeline.plan import camera_at
    W, H = scenes["width"], scenes["height"]
    out = []
    t = shot["start"]
    while t <= shot["end"] + 1e-6:
        x, y, m = camera_at(shot["camera"], t, shot.get("drift_per_s", 0))
        out.append((x - W / 2 * m, y - H / 2 * m, x + W / 2 * m, y + H / 2 * m, m))
        t += step
    return out


def imagery_levels(scenes):
    """Which basemap levels (resolution + area) the camera needs: a coarse base covering every view, plus finer
    levels covering only the views zoomed in enough to need them."""
    vs = [v for s in scenes["shots"] if s["kind"] == "map" for v in views(s, scenes)]
    if not vs:
        return []
    max_m, min_m = max(v[4] for v in vs), min(v[4] for v in vs)
    base = max([r for r in LEVELS if r <= max_m] or [LEVELS[-1]])
    chosen = []
    for r in LEVELS:
        if r > base or r < 10:
            continue
        need = [v for v in vs if v[4] < 2 * r] if r < base else vs
        if r < base and (not need or min_m >= 2 * r):
            continue
        x0, y0 = min(v[0] for v in need), min(v[1] for v in need)
        x1, y1 = max(v[2] for v in need), max(v[3] for v in need)
        need = list(from_3857(x0, y0)) + list(from_3857(x1, y1))  # what the camera actually sees
        pad = 0.12 * max(x1 - x0, y1 - y0)  # downloads get a generous margin so a re-planned camera still fits
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        if (x1 - x0) / r * (y1 - y0) / r > MAX_PX:
            continue  # too big at this resolution: coarser level will do
        lon0, lat0 = from_3857(x0, y0)
        lon1, lat1 = from_3857(x1, y1)
        chosen.append({"res_m": r, "bbox": [lon0, lat0, lon1, lat1], "need": need})
        if r <= min_m:
            break
    return chosen


def grade(src, dst, g):
    """Colour-grade a basemap level for the look: first transfer its Lab mean/std onto the reference's (measured on
    the reels' map pixels, tokens looks.<look>.basemap.grade.color_match), then optional saturation/contrast/brightness."""
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    im = Image.open(src).convert("RGB")
    cm = g.get("color_match")
    if cm:
        lab = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB).astype(np.float32)
        valid = lab[..., 0] > 2  # ignore empty (black) margins
        mu, sd = lab[valid].mean(0), lab[valid].std(0) + 1e-6
        lab = (lab - mu) / sd * np.array(cm["lab_std"], np.float32) + np.array(cm["lab_mean"], np.float32)
        im = Image.fromarray(cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    im = ImageEnhance.Color(im).enhance(g.get("saturation", 1))
    im = ImageEnhance.Contrast(im).enhance(g.get("contrast", 1))
    im = ImageEnhance.Brightness(im).enhance(g.get("brightness", 1))
    im.save(dst, quality=92)


def prepare(ep, scenes_path, tokens):
    from pipeline import imagery
    ep = Path(ep)
    sc = read_json(scenes_path)
    rd = ep / "render"
    (rd / "img").mkdir(parents=True, exist_ok=True)
    (rd / "scenes").mkdir(exist_ok=True)
    for f in (rd / "scenes").glob("*.js"):
        f.unlink()
    look = tokens["looks"][sc["look"]]
    levels, credits = [], set(sc.get("credits", []))
    for lv in imagery_levels(sc):
        path = imagery.for_view(lv["bbox"], lv["res_m"], need=lv["need"])
        meta = imagery.build(lv["bbox"], lv["res_m"], path)
        name = f"img/L{lv['res_m']}_{sha(path.name, look['basemap']['grade'])}.jpg"
        if not (rd / name).exists():
            grade(path.with_suffix(".jpg"), rd / name, look["basemap"]["grade"])
        levels.append({"file": name, "res_m": lv["res_m"], "bounds_3857": meta["bounds_3857"]})
        credits.add(meta["credit"])
    sc["imagery"] = levels
    if any(it["resolved"] and "OpenStreetMap" in it["resolved"]["source"] for it in read_json(ep / "geo.json")["items"]):
        credits.add("© OpenStreetMap contributors")
    sc["credits"] = sorted(credits)
    (rd / "data.js").write_text("const TOKENS = " + json.dumps(tokens, ensure_ascii=False) + ";\nconst SCENES = " +
                                json.dumps(sc, ensure_ascii=False) + ";\n", encoding="utf-8")
    (rd / "config.js").write_text(
        f"const PROJECT = {{ duration: {sc['duration']}, fps: {sc['fps']}, width: {sc['width']}, height: {sc['height']} }};\n",
        encoding="utf-8")
    for i, s in enumerate(sc["shots"]):
        (rd / "scenes" / f"{s['id']}.js").write_text(f"// shot {s['id']} ({s['kind']}, {s['start']}-{s['end']} s)\n"
                                                     f"MAPANI.shot(SCENES.shots[{i}]);\n", encoding="utf-8")
    fonts = ENGINE / "node_modules" / "@fontsource"
    faces = []
    for fam, pkg in (("Inter", "inter"), ("Poppins", "poppins"), ("Playfair Display", "playfair-display")):
        for f in sorted((fonts / pkg / "files").glob(f"{pkg}-latin-*-*.woff2")):
            wt, style = f.stem.split("-")[-2:]
            faces.append(f"@font-face{{font-family:'{fam}';src:url('{f.as_uri()}') format('woff2');"
                         f"font-weight:{wt};font-style:{style};}}")
    scripts = "\n".join(f'<script src="scenes/{s["id"]}.js"></script>' for s in sc["shots"])
    (rd / "studio.html").write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>mapani studio</title>
<style>{''.join(faces)}
html,body{{margin:0;background:#111;color:#eee;font:13px system-ui}} main{{padding:10px;display:grid;gap:8px;justify-items:start}}
#out{{height:86vh;width:auto;aspect-ratio:{sc['width']}/{sc['height']};background:#000}} .bar{{display:flex;gap:10px;width:60vh}} #scrub{{flex:1}}
</style></head><body><main>
<canvas id="out" width="{sc['width']}" height="{sc['height']}"></canvas>
<div class="bar"><input id="scrub" type="range" min="0" max="10" step="0.0333" value="0"><span id="tt">loading...</span></div>
</main>
<script src="config.js"></script>
<script src="data.js"></script>
<script src="{(ENGINE / 'kit.js').as_uri()}"></script>
<script src="{(ENGINE / 'mapani.js').as_uri()}"></script>
{scripts}
</body></html>
""", encoding="utf-8")
    return rd, sc


def node(rd, *args):
    cmd = ["node", str(ENGINE / "render.mjs"), *args]
    r = subprocess.run(cmd, cwd=rd)
    if r.returncode:
        raise SystemExit(f"render failed: {' '.join(cmd)}")


def render(ep, scenes_path=None, workers=3, sheet=None, out=None):
    ep = Path(ep)
    tokens = read_json(ROOT / "style/tokens.json")
    rd, sc = prepare(ep, scenes_path or ep / "scenes.json", tokens)
    fps = str(sc["fps"])
    if sheet:
        node(rd, f"--sheet={sheet}", "--cols=4", "--w=360", f"--fps={fps}", "--out=out/sheet.jpg")
        return rd / "out" / "sheet.jpg"
    frames = rd / "out" / "frames"
    if frames.exists():
        shutil.rmtree(frames)  # scenes may have changed: never mix frames from two plans
    node(rd, "--frames", f"--workers={workers}", f"--fps={fps}")
    # face-cam popups/buttons: transparent PNG frames per shot, composited over the zoomed clip by assemble.py
    for s in sc["shots"]:
        ov = rd / "out" / f"overlay_{s['id']}"
        if ov.exists():
            shutil.rmtree(ov)
        if s["kind"] == "facecam" and s.get("layers"):
            off = sc.get("offset", 0)
            node(rd, "--png", "--overlay", f"--range={s['start'] - off}:{s['end'] - off}", f"--fps={fps}",
                 f"--workers={workers}", f"--out=out/overlay_{s['id']}")
    audio = None
    nar = next(ep.glob("narration.*"), None)
    if nar:
        audio = (rd / "out" / "audio.wav").resolve()  # absolute: render.mjs runs with cwd = the render folder
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(sc.get("offset", 0)), "-t", str(sc["duration"]),
                        "-i", str(nar), str(audio)], check=True)
    out = Path(out or ep / "video_raw.mp4").resolve()
    node(rd, "--encode", f"--fps={fps}", f"--out={out}", *( [f"--audio={audio}"] if audio else []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--scenes")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sheet", help="only a contact sheet at these times, e.g. 0,2,4,6")
    ap.add_argument("--out")
    a = ap.parse_args()
    print(render(a.episode, a.scenes, a.workers, a.sheet, a.out))


if __name__ == "__main__":
    main()
