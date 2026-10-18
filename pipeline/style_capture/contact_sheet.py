"""Annotated contact sheet of a reel: one frame per step, labelled with time, segment kind and the words spoken then.

    python -m pipeline.style_capture.contact_sheet inputs/reference_reels/1.mp4 --measure style/measure/r1.json \
        --transcript style/transcripts/r1.transcript.json --out style/contact_sheets/r1.jpg [--step 1.0] [--cols 8]

Also used later by the review loop on our own output (same layout, so sheets compare side by side).
Text is drawn with PIL so Devanagari renders (OpenCV's Hershey fonts can't).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

KIND_COL = {"map": (40, 160, 255), "facecam": (255, 90, 90), "broll": (255, 200, 40), "other": (40, 160, 255)}
FONT_CANDIDATES = ["C:/Windows/Fonts/Nirmala.ttc", "C:/Windows/Fonts/NirmalaB.ttf", "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"]


def font(size):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def kind_at(segs, t):
    for g in segs:
        if g["start"] <= t < g["end"]:
            return "map" if g["kind"] == "other" else g["kind"]
    return "map"


def sheet(video, segs, words, out, step=1.0, cols=8, tw=200, title=""):
    cap = cv2.VideoCapture(str(video))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    th = round(tw * 16 / 9)
    times = np.arange(0, dur - 0.05, step)
    rows = -(-len(times) // cols)
    cap_h, head = 40, 44
    img = Image.new("RGB", (cols * tw, head + rows * (th + cap_h)), (18, 18, 20))
    d = ImageDraw.Draw(img)
    f_small, f_head = font(14), font(22)
    d.text((10, 10), title, fill=(255, 255, 255), font=f_head)
    for k, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t + step / 2) * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.cvtColor(cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x, y = (k % cols) * tw, head + (k // cols) * (th + cap_h)
        img.paste(Image.fromarray(fr), (x, y))
        kind = kind_at(segs, t + step / 2)
        d.rectangle([x, y + th, x + tw - 1, y + th + 5], fill=KIND_COL[kind])
        said = " ".join(w["w"] for w in words if t <= w["start"] < t + step)
        d.text((x + 4, y + th + 6), f"{t:.0f}s {kind}", fill=(230, 230, 230), font=f_small)
        d.text((x + 4, y + th + 22), said[:34], fill=(170, 200, 255), font=f_small)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--measure")
    ap.add_argument("--transcript")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--segments", help="segments.json (checked by eye) + --key; overrides --measure")
    ap.add_argument("--key")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    segs = json.loads(Path(a.measure).read_text(encoding="utf-8"))["segments"] if a.measure else []
    if a.segments:
        segs = json.loads(Path(a.segments).read_text(encoding="utf-8"))[a.key]
    words = json.loads(Path(a.transcript).read_text(encoding="utf-8"))["words"] if a.transcript else []
    sheet(a.video, segs, words, a.out, a.step, a.cols, title=a.title or Path(a.video).name)


if __name__ == "__main__":
    main()
