"""Camera moves, overlay pop-ins and speech sync, from measure.py output + transcripts.

    python -m pipeline.style_capture.moves --measure style/measure --transcripts style/transcripts --out style/moves.json

Camera moves: inside non-facecam segments, the per-sample log-zoom and pan are smoothed, and a "move" is a run where
the camera speed stays above a threshold. For each move: duration, total zoom factor, total pan, and the easing shape —
the fraction of progress reached at 25/50/75 % of the move's time, compared with linear / easeInOutCubic / easeOutCubic.

Overlay pop-ins: a jump in the share of overlay-palette pixels (yellow / green / white text).

Sync: for each pop-in, the nearest spoken word (start time) within ±1.5 s -> offset = popin_t - word_t
(negative = the graphic leads the word).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

EASES = {
    "linear": lambda x: x,
    "easeInOutCubic": lambda x: 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2,
    "easeOutCubic": lambda x: 1 - (1 - x) ** 3,
    "easeInOutSine": lambda x: -(math.cos(math.pi * x) - 1) / 2,
    "easeInOutQuint": lambda x: 16 * x ** 5 if x < .5 else 1 - (-2 * x + 2) ** 5 / 2,
}


def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode="same")


def camera_moves(samples, segs, dt):
    moves = []
    for g in segs:
        if g["kind"] == "facecam":
            continue
        part = [s for s in samples if g["start"] <= s["t"] < g["end"]]
        if len(part) < 6:
            continue
        lz = np.array([math.log(s["cam"]["zoom"]) if s["cam"] else 0.0 for s in part])
        dx = np.array([s["cam"]["dx"] if s["cam"] else 0.0 for s in part])
        dy = np.array([s["cam"]["dy"] if s["cam"] else 0.0 for s in part])
        # camera speed per sample: zoom rate + pan rate (both ~ "fraction of frame per sample")
        speed = smooth(np.abs(lz) + np.hypot(dx, dy * 16 / 9), 5)
        on = speed > 0.004
        k = 0
        while k < len(part):
            if not on[k]:
                k += 1
                continue
            a = k
            while k < len(part) and on[k]:
                k += 1
            b = k
            if b - a < 4:  # < 0.4 s: jitter, not a move
                continue
            cum_z = np.cumsum(lz[a:b])
            cum_p = np.cumsum(np.hypot(dx[a:b], dy[a:b]))
            prog = cum_z / cum_z[-1] if abs(cum_z[-1]) > 0.05 else cum_p / max(cum_p[-1], 1e-6)
            tt = np.linspace(0, 1, len(prog))
            q = {f"p{int(f*100)}": round(float(np.interp(f, tt, prog)), 3) for f in (.25, .5, .75)}
            fit = {n: round(float(np.mean([(np.interp(x, tt, prog) - e(x)) ** 2 for x in np.linspace(0, 1, 21)])) ** .5, 3)
                   for n, e in EASES.items()}
            moves.append({"start": part[a]["t"], "dur": round((b - a) * dt, 2),
                          "zoom": round(math.exp(float(cum_z[-1])), 3),
                          "pan": [round(float(dx[a:b].sum()), 3), round(float(dy[a:b].sum()), 3)],
                          "progress": q, "ease_rmse": fit, "best_ease": min(fit, key=fit.get)})
    return moves


def popins(samples, segs, thr=0.0025):
    face = [(g["start"], g["end"]) for g in segs if g["kind"] == "facecam"]
    out = []
    for c in ("yellow", "green", "white"):
        v = smooth(np.array([s["text"][c] for s in samples]), 3)
        d = np.diff(v, prepend=v[0])
        last = -9
        for i in range(1, len(v)):
            t = samples[i]["t"]
            if d[i] > thr and d[i] >= d[i - 1] and (i + 1 >= len(d) or d[i] >= d[i + 1]) and t - last > 0.6 \
                    and samples[i]["cut"] < 0.3 and not any(a <= t < b for a, b in face):
                out.append({"t": t, "color": c, "jump": round(float(d[i]), 4)})
                last = t
    return sorted(out, key=lambda x: x["t"])


def sync(pops, words, window=1.5):
    res = []
    for p in pops:
        near = [w for w in words if abs(w["start"] - p["t"]) <= window]
        if not near:
            continue
        # prefer numbers and capitalised / Latin words (stats, place names), else nearest
        def score(w):
            key = any(ch.isdigit() for ch in w["w"]) or (w["w"][:1].isupper())
            return (0 if key else 1, abs(w["start"] - p["t"]))
        w = min(near, key=score)
        res.append({"t": p["t"], "color": p["color"], "word": w["w"], "word_t": w["start"],
                    "offset": round(p["t"] - w["start"], 2)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="style/measure")
    ap.add_argument("--transcripts", default="style/transcripts")
    ap.add_argument("--out", default="style/moves.json")
    a = ap.parse_args()
    report = {}
    for mf in sorted(Path(a.measure).glob("r*.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        name = mf.stem
        tr = json.loads((Path(a.transcripts) / f"{name}.transcript.json").read_text(encoding="utf-8"))
        dt = 1 / m["sample_fps"]
        mv = camera_moves(m["samples"], m["segments"], dt)
        pp = popins(m["samples"], m["segments"])
        sy = sync(pp, tr["words"])
        offs = [s["offset"] for s in sy]
        report[name] = {"moves": mv, "popins": pp, "sync": sy,
                        "sync_median_offset": round(float(np.median(offs)), 2) if offs else None,
                        "segments": m["segments"]}
        print(f"{name}: {len(mv)} moves, {len(pp)} pop-ins, sync median {report[name]['sync_median_offset']}s")
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
