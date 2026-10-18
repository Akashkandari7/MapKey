"""Colour measurements of the map segments of each reference reel.

    python -m pipeline.style_capture.palette --out style/palette.json

Per reel, from frames every 0.5 s inside non-facecam segments (style/measure/<r>.json segments):
    accents    k-means (k=10) over saturated, bright pixels -> hex + share: highlight fills, routes, text colours
    basemap    stats of the unsaturated rest: mean/median brightness and saturation, dominant dark tones (k=5)
    vignette   mean brightness in 6 concentric rings, centre -> corners, relative to the centre
    tint       mean hue/sat of the basemap (tells "dark muted satellite" from "bright satellite" from "greyscale")
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans


def hexc(bgr):
    b, g, r = [int(round(x)) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def frames(video, segs, every=0.5, w=360):
    cap = cv2.VideoCapture(str(video))
    for g in segs:
        if g["kind"] == "facecam":
            continue
        t = g["start"] + 0.25
        while t < g["end"] - 0.1:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if ok:
                yield t, cv2.resize(f, (w, round(w * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)
            t += every
    cap.release()


def analyse(video, segs):
    acc, base, rings = [], [], []
    for _, f in frames(video, segs):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        m = (s > 120) & (v > 120)
        acc.append(f[m][::3])
        base.append(np.column_stack([f[~m][::25], hsv[~m][::25]]))
        h, w = v.shape
        yy, xx = np.mgrid[:h, :w]
        rr = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
        rings.append([float(v[(rr >= a) & (rr < a + 1 / 6)].mean()) for a in np.arange(0, 1, 1 / 6)])
    acc = np.concatenate(acc) if acc else np.zeros((0, 3))
    base = np.concatenate(base)
    out = {}
    if len(acc) > 200:
        km = KMeans(10, n_init=4, random_state=0).fit(acc[:60000].astype(np.float32))
        share = np.bincount(km.labels_, minlength=10) / len(km.labels_)
        order = np.argsort(-share)
        out["accents"] = [{"hex": hexc(km.cluster_centers_[i]), "share": round(float(share[i]), 3)} for i in order]
        out["accent_pixel_share"] = round(len(acc) * 3 / (len(base) * 25 + len(acc) * 3), 4)
    bgr, hsv = base[:, :3].astype(np.float32), base[:, 3:].astype(np.float32)
    km = KMeans(5, n_init=4, random_state=0).fit(bgr[:80000])
    share = np.bincount(km.labels_, minlength=5) / len(km.labels_)
    out["basemap"] = {"brightness_mean": round(float(hsv[:, 2].mean() / 255), 3),
                      "brightness_median": round(float(np.median(hsv[:, 2]) / 255), 3),
                      "saturation_mean": round(float(hsv[:, 1].mean() / 255), 3),
                      "tones": [{"hex": hexc(km.cluster_centers_[i]), "share": round(float(share[i]), 3)}
                                for i in np.argsort(-share)]}
    r = np.mean(rings, axis=0)
    out["vignette_rel"] = [round(float(x / r[0]), 3) for x in r]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="style/measure")
    ap.add_argument("--videos", default="inputs/reference_reels")
    ap.add_argument("--out", default="style/palette.json")
    a = ap.parse_args()
    res = {}
    for mf in sorted(Path(a.measure).glob("r*.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        name = mf.stem
        res[name] = analyse(Path(a.videos) / f"{name[1:]}.mp4", m["segments"])
        p = res[name]
        print(name, "base", p["basemap"]["brightness_mean"], p["basemap"]["saturation_mean"],
              "vig", p["vignette_rel"], "acc", [x["hex"] for x in p.get("accents", [])][:8])
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
