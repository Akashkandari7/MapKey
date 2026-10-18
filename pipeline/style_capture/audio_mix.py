"""Sound measurements of each reference reel: loudness, music bed under the voice, SFX hits.

    python -m pipeline.style_capture.audio_mix --out style/audio.json

    loudness     integrated LUFS, loudness range, true peak (ffmpeg ebur128 on the original stereo track)
    speech_db    mean RMS dBFS inside spoken words (transcript word spans)
    gap_db       mean RMS dBFS in pauses >= 0.35 s between words (no voice) -> the music bed level
    bed_below    speech_db - gap_db: how far the music sits under the voice
    gap_has_music  pauses that are not near-silent (gap RMS > -50 dBFS) -> a continuous bed is present
    hits         short broadband transients (onset strength peaks) in non-speech-dominated spots -> SFX candidates,
                 with the camera move / segment boundary they line up with (from style/moves.json)
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import librosa
import numpy as np

SR = 22050


def ebur128(path):
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    get = lambda k: float(re.search(k + r":\s+(-?[\d.]+)", tail).group(1))
    return {"integrated_lufs": get("I"), "lra": get("LRA"), "true_peak_dbfs": get("Peak")}


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def analyse(video, transcript, moves):
    y, _ = librosa.load(str(video), sr=SR, mono=True)
    words = transcript["words"]
    sp = np.concatenate([y[int(w["start"] * SR):int(w["end"] * SR)] for w in words]) if words else y[:0]
    gaps = [(a["end"], b["start"]) for a, b in zip(words, words[1:]) if b["start"] - a["end"] >= 0.35]
    gap_db = [db(y[int((a + .05) * SR):int((b - .05) * SR)]) for a, b in gaps]
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=256)
    peaks = librosa.util.peak_pick(onset, pre_max=8, post_max=8, pre_avg=40, post_avg=40, delta=np.percentile(onset, 97) * .5, wait=20)
    times = librosa.frames_to_time(peaks, sr=SR, hop_length=256)
    in_word = lambda t: any(w["start"] - .03 <= t <= w["end"] + .03 for w in words)
    move_starts = [m["start"] for m in moves.get("moves", [])]
    bounds = [g["start"] for g in moves.get("segments", [])][1:]
    hits = []
    for t, p in zip(times, peaks):
        if in_word(t):
            continue
        near_move = min((abs(t - s) for s in move_starts), default=9)
        near_cut = min((abs(t - s) for s in bounds), default=9)
        hits.append({"t": round(float(t), 2), "strength": round(float(onset[p]), 2),
                     "near_move_s": round(near_move, 2), "near_cut_s": round(near_cut, 2)})
    s_db = db(sp) if len(sp) else None
    g = float(np.mean(gap_db)) if gap_db else None
    return {"loudness": ebur128(video), "speech_db": round(s_db, 1) if s_db else None,
            "gap_db": round(g, 1) if g else None, "bed_below_db": round(s_db - g, 1) if s_db and g else None,
            "gaps": len(gaps), "gap_has_music": round(float(np.mean([x > -50 for x in gap_db])), 2) if gap_db else None,
            "hits": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="inputs/reference_reels")
    ap.add_argument("--transcripts", default="style/transcripts")
    ap.add_argument("--moves", default="style/moves.json")
    ap.add_argument("--out", default="style/audio.json")
    a = ap.parse_args()
    mv = json.loads(Path(a.moves).read_text(encoding="utf-8"))
    res = {}
    for tf in sorted(Path(a.transcripts).glob("r*.transcript.json")):
        name = tf.name.split(".")[0]
        r = analyse(Path(a.videos) / f"{name[1:]}.mp4", json.loads(tf.read_text(encoding="utf-8")), mv.get(name, {}))
        res[name] = r
        near = [h for h in r["hits"] if min(h["near_move_s"], h["near_cut_s"]) < .35]
        print(f"{name}: {r['loudness']}, speech {r['speech_db']} dB, gaps {r['gap_db']} dB "
              f"(bed {r['bed_below_db']} dB below, music in {r['gap_has_music']} of gaps), "
              f"{len(r['hits'])} hits, {len(near)} near a move/cut")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
