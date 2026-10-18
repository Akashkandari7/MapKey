"""Per-frame measurements of a reference reel, for STYLE.md.

    python -m pipeline.style_capture.measure inputs/reference_reels/1.mp4 --out style/measure/r1.json [--fps 10]

For every sampled frame (default 10 fps, analysed at 360x640):
    cut     histogram distance to the previous sample (hard cuts spike)
    face    largest frontal face area / frame area (face-cam segments have a big face)
    cam     similarity transform from the previous sample, from ORB tracks + RANSAC: zoom (scale ratio), pan dx/dy
            (fraction of frame width/height), rotation (deg), inlier count. Overlays that move on their own are
            outliers; a static frame gives zoom 1, pan 0.
    text    pixel share of the overlay palette (saturated yellow, neon green, near-white) -> pop-ins show as jumps
    sat/val mean saturation / brightness
Then groups samples into segments (cuts + face/no-face changes) and labels each one facecam | map | other.
"map" vs "broll" can't be told apart reliably by pixels alone, so "other" segments get checked by eye.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

AW, AH = 360, 640
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ORB = cv2.ORB_create(1500, fastThreshold=12)
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def text_masks(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (h >= 22) & (h <= 34) & (s > 150) & (v > 200)
    green = (h >= 40) & (h <= 75) & (s > 150) & (v > 190)
    white = (s < 25) & (v > 235)
    return yellow, green, white


def cam_motion(prev_g, g):
    k1, d1 = ORB.detectAndCompute(prev_g, None)
    k2, d2 = ORB.detectAndCompute(g, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    m = BF.match(d1, d2)
    if len(m) < 20:
        return None
    p1 = np.float32([k1[x.queryIdx].pt for x in m])
    p2 = np.float32([k2[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=3000)
    if M is None or inl is None or inl.sum() < 15:
        return None
    s = math.hypot(M[0, 0], M[1, 0])
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    # pan of the frame centre, in fractions of the frame (camera moving right = content moving left = negative dx)
    cx, cy = AW / 2, AH / 2
    nx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    ny = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    return {"zoom": round(s, 5), "dx": round((nx - cx) / AW, 5), "dy": round((ny - cy) / AH, 5),
            "rot": round(rot, 4), "inl": int(inl.sum())}


def measure(path, fps=10):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps))
    out, prev_hist, prev_g, i = [], None, None, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        ok, f = cap.retrieve()
        if not ok:
            break
        t = i / src_fps
        f = cv2.resize(f, (AW, AH), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = 0.0 if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        faces = FACE.detectMultiScale(g, 1.15, 5, minSize=(60, 60))
        face = max((w * h for (x, y, w, h) in faces), default=0) / (AW * AH)
        yl, gr, wh = text_masks(hsv)
        rec = {"t": round(t, 3), "cut": round(cut, 4), "face": round(face, 4),
               "text": {"yellow": round(float(yl.mean()), 5), "green": round(float(gr.mean()), 5),
                        "white": round(float(wh.mean()), 5)},
               "sat": round(float(hsv[..., 1].mean()) / 255, 4), "val": round(float(hsv[..., 2].mean()) / 255, 4),
               "cam": cam_motion(prev_g, g) if prev_g is not None and cut < 0.35 else None}
        out.append(rec)
        prev_hist, prev_g, i = hist, g, i + 1
    cap.release()
    return {"src": str(path), "src_fps": src_fps, "frames": n, "sample_fps": src_fps / step, "samples": out,
            "segments": segment(out)}


def segment(s, cut_thr=0.45, face_thr=0.04):
    """Split at hard cuts and at face/no-face changes (smoothed over 5 samples); label each run."""
    if not s:
        return []
    face_on = np.convolve([x["face"] > face_thr for x in s], np.ones(5) / 5, mode="same") > 0.5
    bounds = [0]
    for k in range(1, len(s)):
        if s[k]["cut"] > cut_thr or face_on[k] != face_on[k - 1]:
            if k - bounds[-1] >= 3:  # ignore flicker shorter than 0.3 s
                bounds.append(k)
    bounds.append(len(s))
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        part = s[a:b]
        cams = [x["cam"] for x in part if x["cam"]]
        zoom_total = float(np.prod([c["zoom"] for c in cams])) if cams else 1.0
        segs.append({"start": part[0]["t"], "end": s[b]["t"] if b < len(s) else round(part[-1]["t"] + 0.1, 3),
                     "kind": "facecam" if face_on[a:b].mean() > 0.5 else "other",
                     "zoom_total": round(zoom_total, 3),
                     "pan_total": [round(sum(c["dx"] for c in cams), 3), round(sum(c["dy"] for c in cams), 3)],
                     "tracked": round(len(cams) / max(1, len(part)), 2)})
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10)
    a = ap.parse_args()
    r = measure(a.video, a.fps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r), encoding="utf-8")
    print(f"{a.video}: {len(r['samples'])} samples, {len(r['segments'])} segments")
    for g in r["segments"]:
        print(f"  {g['start']:6.2f}-{g['end']:6.2f}  {g['kind']:8s} zoom x{g['zoom_total']:<6} pan {g['pan_total']}")


if __name__ == "__main__":
    main()
