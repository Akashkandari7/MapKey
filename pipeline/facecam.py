"""Where in the narration the creator's face-cam can be used.

    python -m pipeline.facecam --clip inputs/ep/facecam.mp4 --narration work/ep/narration.wav

A raw face-cam recording shows the face all the way through. An already-edited video (the creator's own reel, used
as the source for a remake) shows the face only in parts, between cartoons, maps and screenshots. Both work: frames
are sampled every 0.25 s, faces found (OpenCV), and the face-cam's own look (its studio background) learned from the
face frames that look most alike; a frame counts as face-cam when it has that look. A cartoon face does not.

Returns windows in NARRATION time (the clip is placed by its sound, assemble.clip_offset).
"""
import argparse
import json

import cv2
import numpy as np

STEP = 0.25


def _hist(fr):
    h = cv2.calcHist([cv2.cvtColor(cv2.resize(fr, (180, 320)), cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 16],
                     [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def windows(clip, offset=0.0, min_len=0.75, corr=0.8):
    cap = cv2.VideoCapture(str(clip))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1e-6, cap.get(cv2.CAP_PROP_FPS))
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    samples = []
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = [f for f in cas.detectMultiScale(g, 1.1, 6, minSize=(90, 90)) if f[2] > fr.shape[1] * 0.12]
        samples.append((t, bool(len(faces)), _hist(fr)))
        t += STEP
    face = [s for s in samples if s[1]]
    if not face:
        return []
    # the face-cam's look: the face frame most similar to all other face frames (cartoon faces are the minority)
    H = np.array([s[2] for s in face])
    sims = np.array([[cv2.compareHist(a, b, cv2.HISTCMP_CORREL) for b in H] for a in H])
    ref = H[int(sims.mean(1).argmax())]
    look = [cv2.compareHist(ref, s[2], cv2.HISTCMP_CORREL) > corr for s in samples]
    out, cur = [], None
    for (t, _, _), on in zip(samples, look):
        if on and cur is None:
            cur = [t, t]
        elif on:
            cur[1] = t
        elif cur is not None:
            if cur[1] - cur[0] >= min_len:
                out.append([round(cur[0] + offset, 2), round(cur[1] + STEP + offset, 2)])
            cur = None
    if cur and cur[1] - cur[0] >= min_len:
        out.append([round(cur[0] + offset, 2), round(cur[1] + STEP + offset, 2)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--narration")
    a = ap.parse_args()
    off = 0.0
    if a.narration:
        from pipeline.assemble import clip_offset
        off = clip_offset(a.clip, a.narration)[0]
    print(json.dumps({"offset": round(off, 3), "windows": windows(a.clip, off)}))


if __name__ == "__main__":
    main()
