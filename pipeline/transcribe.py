"""Stage 1: narration audio -> transcript.json with word-level timestamps.

    python -m pipeline.transcribe <audio> [<audio> ...] --out work/<episode>/transcript.json [--lang hi]

Output:
    {
      "audio": "...", "duration": 51.2, "language": "hi", "language_prob": 0.97,
      "models": ["large-v3", "large-v2"], "device": "cuda/float16",
      "repaired_spans": [[13.6, 30.0]],          # stretches where the primary model failed and the fallback's words were used
      "quality": {"words": 176, "words_per_s": 3.48, "min_words_per_3s": 6, "long_words": 0},
      "segments": [{"start", "end", "text", "words": [{"w", "start", "end", "p"}], "model"}],
      "words":    [{"w", "start", "end", "p", "seg"}]          # flat list, what later stages use
    }

Why two models. On this fast, pause-free Hinglish narration whisper sometimes stops transcribing part-way through a window
and stretches the remaining words over the gap. Measured on the six reference reels (temperature 0, deterministic):
large-v3 is right on r1 (176 words) but loses most of r3/r6 (89/60 words); large-v2 is right on r3/r6 (233/177) but
loses part of r1 (136). So: transcribe with large-v3, find the stretches that look broken (too few words for continuous
speech, or words lasting over a second), and only if there are any, transcribe with large-v2 and take its words there.

Other settings, also from those tests: no initial_prompt (it made large-v3 drop half of r1); VAD chunks capped at 12 s;
temperature 0 so the same audio always gives the same transcript.

Hindi/Hinglish: whisper writes Hindi speech in Devanagari and English words (km, metro, stations) usually in Latin.
Later stages (entities via LLM) read either script. Captions are not burned word-by-word (STYLE.md), so the script is
left as whisper writes it.
"""
import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pipeline import gpu  # noqa: F401  (adds CUDA DLL dirs before ctranslate2 loads)

SR = 16000
PRIMARY, FALLBACK = "large-v3", "large-v2"
MAX_CHUNK_S = 12
WIN_S, MIN_WORDS_PER_WIN = 3.0, 3   # continuous speech runs ~3 words/s; fewer than 3 in 3 s means words were dropped
LONG_WORD_S = 1.0                   # a "word" longer than this is a stretched timestamp over dropped speech
LOUD_DB = -35                       # windows quieter than this are real silence, not failures


def audio_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def decode_16k(path):
    """Any audio/video file -> mono 16 kHz float32, via ffmpeg (avoids PyAV version drift inside faster-whisper)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)


def load_model(name):
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(name, device="cuda", compute_type="float16"), "cuda/float16"
    except Exception as e:  # no GPU / out of VRAM -> CPU int8 (slower, same output format)
        print(f"[transcribe] CUDA unavailable ({e.__class__.__name__}: {e}); falling back to CPU int8", file=sys.stderr)
        return WhisperModel(name, device="cpu", compute_type="int8"), "cpu/int8"


def run_model(model, pcm, lang, name):
    segs, info = model.transcribe(
        pcm, language=lang, word_timestamps=True, temperature=0.0, beam_size=5, condition_on_previous_text=False,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 300, "max_speech_duration_s": MAX_CHUNK_S})
    out = []
    for s in segs:
        ws = [{"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3), "p": round(w.probability, 3)}
              for w in (s.words or []) if w.word.strip()]
        if ws:
            out.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(), "words": ws,
                        "model": name})
    return out, info


def bad_spans(pcm, segments):
    """Loud stretches with too few words, or with stretched words -> [(start, end)], merged."""
    words = [w for s in segments for w in s["words"]]
    dur = len(pcm) / SR
    starts = np.array([w["start"] for w in words]) if words else np.zeros(0)
    spans = []
    for a in np.arange(0, max(dur - WIN_S, 0) + 1e-6, WIN_S / 2):
        chunk = pcm[int(a * SR):int((a + WIN_S) * SR)]
        loud = 20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-9) > LOUD_DB
        if loud and ((starts >= a) & (starts < a + WIN_S)).sum() < MIN_WORDS_PER_WIN:
            spans.append([a, min(a + WIN_S, dur)])
    spans += [[w["start"], w["end"]] for w in words if w["end"] - w["start"] > LONG_WORD_S]
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 0.2:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def quality(words, dur):
    starts = np.array([w["start"] for w in words]) if words else np.zeros(0)
    wins = [int(((starts >= a) & (starts < a + WIN_S)).sum()) for a in np.arange(0, max(dur - WIN_S, 0) + 1e-6, WIN_S / 2)]
    span = (words[-1]["end"] - words[0]["start"]) if words else 0
    return {"words": len(words), "words_per_s": round(float(len(words) / span), 2) if span else 0,
            "min_words_per_3s": min(wins) if wins else 0,
            "long_words": sum(1 for w in words if w["end"] - w["start"] > LONG_WORD_S)}


def splice(primary, fallback, spans):
    """Replace primary words inside each span with the fallback's words there (spans widened to word edges)."""
    def inside(w, a, b):
        return w["end"] > a and w["start"] < b
    out = []
    for s in primary:
        keep = [w for w in s["words"] if not any(inside(w, a, b) for a, b in spans)]
        if keep:
            out.append(dict(s, words=keep, start=keep[0]["start"], end=keep[-1]["end"],
                            text=" ".join(w["w"] for w in keep)))
    for s in fallback:
        take = [w for w in s["words"] if any(inside(w, a, b) for a, b in spans)]
        if take:
            out.append(dict(s, words=take, start=take[0]["start"], end=take[-1]["end"],
                            text=" ".join(w["w"] for w in take)))
    out.sort(key=lambda s: s["start"])
    return out


class Transcriber:
    """Holds at most one whisper model on the GPU at a time (6 GB laptop GPU: never both)."""

    def __init__(self):
        self.name, self.model, self.device = None, None, None

    def get(self, name):
        if self.name != name:
            self.model = None
            gc.collect()
            self.model, self.device = load_model(name)
            self.name = name
        return self.model

    def __call__(self, audio, lang=None):
        pcm = decode_16k(audio)
        dur = len(pcm) / SR
        segments, info = run_model(self.get(PRIMARY), pcm, lang, PRIMARY)
        lang = lang or info.language
        spans, models = bad_spans(pcm, segments), [PRIMARY]
        if spans:
            fb, _ = run_model(self.get(FALLBACK), pcm, lang, FALLBACK)
            fb_words = [w for s in fb for w in s["words"]]
            # only take the fallback where it is actually better (more words in that span)
            spans = [sp for sp in spans
                     if sum(1 for w in fb_words if sp[0] <= w["start"] < sp[1]) >
                     sum(1 for s in segments for w in s["words"] if sp[0] <= w["start"] < sp[1])]
            if spans:
                segments = splice(segments, fb, spans)
                models.append(FALLBACK)
        words = [dict(w, seg=i) for i, s in enumerate(segments) for w in s["words"]]
        q = quality(words, dur)
        # one drawn-out word ("insaaane") is normal speech; only too-few-words windows mean speech is missing
        if q["min_words_per_3s"] < MIN_WORDS_PER_WIN:
            print(f"[transcribe] WARNING {audio}: still looks incomplete after repair {q} - check the transcript",
                  file=sys.stderr)
        return {"audio": str(audio), "duration": round(audio_duration(audio), 3), "language": info.language,
                "language_prob": round(info.language_probability, 3), "models": models, "device": self.device,
                "repaired_spans": [[round(float(a), 2), round(float(b), 2)] for a, b in spans], "quality": q,
                "segments": segments, "words": words}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--out", help="output json (single input) or directory (several inputs)")
    ap.add_argument("--lang", default=None, help="force a language code, e.g. hi; default auto-detect")
    a = ap.parse_args()
    tr = Transcriber()
    for src in a.audio:
        out = Path(a.out) if a.out and len(a.audio) == 1 else Path(a.out or ".") / (Path(src).stem + ".transcript.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        t = tr(src, a.lang)
        out.write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[transcribe] {src}: {t['quality']}, models {t['models']}, repaired {t['repaired_spans']} -> {out}")


if __name__ == "__main__":
    main()
