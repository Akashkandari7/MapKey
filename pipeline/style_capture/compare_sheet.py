"""Side-by-side sheet: reference reel frames (top row) vs our render (bottom row) at the same times.

    python -m pipeline.style_capture.compare_sheet --ref inputs/reference_reels/1.mp4 --ours work/ep/render/out/sheet_frames \
        --times 0.3,1.5,2.6 --out work/ep/compare.jpg

`--ours` is a video file, or the contact sheet's own frames rendered as stills by render.mjs --stills.
Used by the review loop and when tuning tokens against a reference.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def grab(path, t, w):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read(); cap.release()
    if not ok:
        return np.zeros((round(w * 16 / 9), w, 3), np.uint8)
    return cv2.resize(f, (w, round(w * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)


def still(folder, t, w):
    f = Path(folder) / f"t{t:.2f}".replace(".", "_")
    img = cv2.imread(str(f) + ".png")
    return cv2.resize(img, (w, round(w * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--times", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=270)
    a = ap.parse_args()
    ts = [float(x) for x in a.times.split(",")]
    get = (lambda t: still(a.ours, t, a.w)) if Path(a.ours).is_dir() else (lambda t: grab(a.ours, t, a.w))
    top = np.hstack([grab(a.ref, t, a.w) for t in ts])
    bot = np.hstack([get(t) for t in ts])
    lab = np.zeros((26, top.shape[1], 3), np.uint8)
    for i, t in enumerate(ts):
        cv2.putText(lab, f"{t:.1f}s  ref (top) / ours", (i * a.w + 6, 18), 0, 0.5, (255, 255, 255), 1)
    cv2.imwrite(a.out, np.vstack([lab, top, bot]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(a.out)


if __name__ == "__main__":
    main()
