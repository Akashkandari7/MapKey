"""Story mode sound: a music bed (Gemini Lyria, cached per episode) and transition effects (synthesised here), mixed
under the narration by assemble.py.

    python -m pipeline.sound --episode work/<ep>        # writes work/<ep>/assemble/music.wav and sfx.wav

Measured on q1-q4 (tokens story.audio): a continuous low, tense bed under the voice and sub-bass booms / whooshes on
the big transitions. The effects are synthesised (filtered noise sweeps, pitched-down sine thumps) so there is no
licence question; the music comes from Lyria, generated for this reel (the user's Gemini key).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from pipeline.common import ROOT, read_json, require_env, sha

SR = 48000


def _env(n, attack, release, curve=2.0):
    a = max(1, int(attack * SR))
    e = np.ones(n)
    e[:a] = np.linspace(0, 1, a) ** curve
    r = max(1, int(release * SR))
    e[-r:] *= np.linspace(1, 0, r) ** curve
    return e


def _sweep_filter(x, f0, f1, q=1.2):
    """Band-pass whose centre glides from f0 to f1 (Hz) over the sound: a state-variable filter, sample by sample."""
    y = np.zeros_like(x)
    low = band = 0.0
    fs = np.geomspace(f0, f1, len(x))
    for i, v in enumerate(x):
        f = 2 * np.sin(np.pi * min(fs[i], SR / 6) / SR)
        high = v - low - band / q
        band += f * high
        low += f * band
        y[i] = band
    return y


def whoosh(dur=0.9, seed=1):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    peak = 0.62  # loudest just after the middle: it lands on the cut
    up = _sweep_filter(noise, 260, 4200)
    env = np.interp(np.arange(n), [0, peak * n, n - 1], [0, 1, 0]) ** 1.6
    mono = up * env
    pan = np.linspace(-0.7, 0.7, n)  # passes left to right
    st = np.stack([mono * np.sqrt((1 - pan) / 2), mono * np.sqrt((1 + pan) / 2)])
    return st / (np.abs(st).max() + 1e-9), peak * dur


def boom(dur=2.4, seed=2):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 32 + 34 * np.exp(-t * 7)  # 66 -> 32 Hz drop
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 2.2)
    click = _sweep_filter(rng.standard_normal(int(0.08 * SR)), 1800, 200) * np.exp(-np.arange(int(0.08 * SR)) / SR * 40)
    body[:len(click)] += 0.35 * click / (np.abs(click).max() + 1e-9)
    body = np.tanh(body * 1.8)  # a little saturation: heard on phone speakers
    # short dark tail (decaying filtered noise)
    tail = _sweep_filter(rng.standard_normal(n), 400, 90) * np.exp(-t * 1.6) * 0.25
    mono = body + tail
    st = np.stack([mono, mono])
    return st / (np.abs(st).max() + 1e-9), 0.0


def riser(dur=1.6, seed=3):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = _sweep_filter(rng.standard_normal(n), 300, 6000)
    tone = np.sin(2 * np.pi * np.cumsum(np.geomspace(180, 900, n)) / SR) * 0.35
    env = (t / dur) ** 2.5
    mono = (noise + tone) * env
    st = np.stack([mono, mono])
    return st / (np.abs(st).max() + 1e-9), dur  # ends on the cut


SFX = {"whoosh": whoosh, "boom": boom, "riser": riser}
AUTO = {"zoom": "whoosh", "whip": "whoosh", "flash": "boom", "glitch": "whoosh"}


def sfx_track(sc, tokens):
    """Every shot's effect at its cut: the storyboard's choice, else one implied by a dramatic transition."""
    A = tokens["story"]["audio"]
    n = int((sc["duration"] + 3) * SR)
    out = np.zeros((2, n))
    for i, s in enumerate(sc["shots"]):
        kind = s.get("sfx") or AUTO.get((s.get("trans") or {}).get("type"))
        if not kind or kind not in SFX or i == 0 and kind != "boom":
            continue
        snd, hit = SFX[kind](seed=i + 7)
        gain = 10 ** (A[f"{kind}_db"] / 20)
        at = int((s["start"] - sc.get("offset", 0) - hit) * SR)
        a, b = max(0, at), min(n, at + snd.shape[1])
        if b > a:
            out[:, a:b] += gain * snd[:, a - at:b - at]
    return out[:, :int(sc["duration"] * SR)]


def music(ep, duration, topic, model="lyria-3-pro-preview"):
    """A bed for this reel from Lyria, cached in work/<ep>/cache; looped with a crossfade if it comes back short."""
    ep = Path(ep)
    prompt = ("Instrumental underscore for an Indian news explainer reel about: " + (topic or "current affairs") +
              ". Dark, tense, modern cinematic documentary style: deep pulsing synth bass, low drones, soft ticking "
              "percussion, subtle risers, steady 95 BPM, no vocals, no lead melody, consistent energy (it sits under a "
              f"voice), about {int(duration) + 8} seconds.")
    key = sha(prompt, model)
    mp3 = ep / "cache" / f"music_{key}.mp3"
    if not mp3.exists():
        import time
        from google import genai
        client = genai.Client(api_key=require_env("GEMINI_API_KEY", "put GEMINI_API_KEY=... in .env"))
        data = None
        for k in range(3):  # a dropped connection must not sink the whole mix
            try:
                r = client.models.generate_content(model=model, contents=[prompt])
                data = next((p.inline_data.data for p in r.candidates[0].content.parts if p.inline_data), None)
                if data:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[sound] music attempt {k + 1} failed: {str(e)[:160]}")
                time.sleep(5 * (k + 1))
        if not data:
            print("[sound] WARNING: no music bed (Lyria unreachable) - the reel gets voice + effects only")
            return None
        mp3.parent.mkdir(parents=True, exist_ok=True)
        mp3.write_bytes(data)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp3), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).T.astype(np.float64)
    need = int(duration * SR)
    xf = int(3 * SR)
    while x.shape[1] < need:  # loop: the tail fades into the head (3 s equal-power crossfade)
        f = np.linspace(0, np.pi / 2, xf)
        mid = x[:, -xf:] * np.cos(f) + x[:, :xf] * np.sin(f)
        x = np.concatenate([x[:, :-xf], mid, x[:, xf:]], axis=1)
    x = x[:, :need]
    fade = int(2.5 * SR)
    x[:, -fade:] *= np.linspace(1, 0, fade)
    return x


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", str(path)],
                       input=x.T.astype(np.float32).tobytes(), check=True)
    return p


def build(ep, sc=None, tokens=None):
    ep = Path(ep)
    tokens = tokens or read_json(ROOT / "style/tokens.json")
    sc = sc or read_json(ep / "scenes.json")
    topic = read_json(ep / "storyboard.json").get("topic") if (ep / "storyboard.json").exists() else None
    out = {}
    m = music(ep, sc["duration"], topic)
    if m is not None:
        m = m / (np.abs(m).max() + 1e-9) * 10 ** (tokens["story"]["audio"]["music_db"] / 20) * 3.0
        write_wav(ep / "assemble" / "music.wav", m)
        out["music"] = ep / "assemble" / "music.wav"
    fx = sfx_track(sc, tokens)
    write_wav(ep / "assemble" / "sfx.wav", fx)
    out["sfx"] = ep / "assemble" / "sfx.wav"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    a = ap.parse_args()
    print(json.dumps({k: str(v) for k, v in build(a.episode).items()}))


if __name__ == "__main__":
    main()
