"""Stages 6-8: rendered picture + face-cam clip + narration -> output/<episode>.mp4 (Instagram-ready).

    python -m pipeline.assemble --episode work/<ep> [--facecam inputs/audio/ep12_facecam.mp4] [--music file] [--out output/ep.mp4]

Face-cam: the clip is placed by its own sound. Its audio is cross-correlated with the narration to find where the clip
sits in the narration timeline (the creator records the outro with the same voice), so every face-cam shot in
scenes.json shows exactly the matching moment, lips in sync, without anyone typing timestamps. Without a clip, the
shot keeps the "FACE-CAM CLIP HERE" placeholder the renderer drew.

Audio: narration normalised to the reference loudness (tokens audio: -14 LUFS integrated, true peak <= -1 dBTP; two-pass
loudnorm, linear). Optional music bed under it, ducked by the voice (sidechain), level from tokens; off by default.

Export: 1080x1920, 30 fps, H.264 High (crf 18, capped 20 Mb/s), AAC 48 kHz 256 kb/s, +faststart. Also writes
output/<ep>.scenes.json and output/<ep>.sheet.jpg (contact sheet) for quick review.
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from pipeline.common import ROOT, read_json, write_json

SR = 8000


def pcm(path, sr=SR):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)


def clip_offset(clip, narration):
    """Seconds to add to clip time to get narration time (narr_t = clip_t + off), by FFT cross-correlation of the
    two soundtracks' onset envelopes. Returns (off, confidence 0..1)."""
    a, b = pcm(narration), pcm(clip)
    if len(b) < SR:
        return None, 0.0

    def env(x):  # 100 Hz energy envelope, mean-removed: robust to different mics / levels
        hop = SR // 100
        e = np.sqrt(np.add.reduceat(x[: len(x) // hop * hop] ** 2, np.arange(0, len(x) // hop * hop, hop)) / hop)
        e = np.log1p(e * 100)
        return (e - e.mean()) / (e.std() + 1e-9)
    ea, eb = env(a), env(b)
    n = 1 << int(np.ceil(np.log2(len(ea) + len(eb))))
    xc = np.fft.irfft(np.fft.rfft(ea, n) * np.conj(np.fft.rfft(eb, n)), n)
    lag = int(np.argmax(xc))
    if lag > n // 2:
        lag -= n
    peak = xc.max() / min(len(ea), len(eb))
    return lag / 100.0, float(np.clip(peak, 0, 1))


def face_center(clip, t0, t1, n=7):
    """Median centre of the largest face in the clip between t0 and t1 (clip time), as fractions of the frame."""
    import cv2
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(clip))
    pts = []
    for t in np.linspace(t0, max(t0, t1 - 0.05), n):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t) * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(f, (360, round(360 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(g, 1.15, 5, minSize=(50, 50))
        if len(faces):
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            pts.append(((x + w / 2) / g.shape[1], (y + h / 2) / g.shape[0]))
    cap.release()
    if not pts:
        return 0.5, 0.4
    a = np.array(pts)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


def zoom_filter(shot, clip, clip_off, W, H, fps, F):
    """zoompan expression for a face-cam shot: punch-in cuts (scale steps) with a slow push between them, each
    segment centred on the face (face at F['face_y'] of the frame height). Times are relative to the shot start."""
    dur = shot["end"] - shot["start"]
    punches = shot.get("punches") or [{"t": 0.0, "scale": 1.0}]
    segs = []
    for i, p in enumerate(punches):
        t0, t1 = p["t"], (punches[i + 1]["t"] if i + 1 < len(punches) else dur)
        fx, fy = face_center(clip, shot["start"] + t0 - clip_off, shot["start"] + t1 - clip_off)
        segs.append((t0, t1, p["scale"], fx, fy))
    push = F["push_per_shot"]

    def nest(f):  # if(lt(it,t1), seg0, if(lt(it,t2), seg1, ...))
        expr = f(segs[-1])
        for s in reversed(segs[:-1]):
            expr = f"if(lt(it,{s[1]:.3f}),{f(s)},{expr})"
        return expr
    z = nest(lambda s: f"{s[2]:.4f}*(1+{push:.4f}*(it-{s[0]:.3f})/{max(0.5, s[1] - s[0]):.3f})")
    fxs = nest(lambda s: f"{s[3]:.4f}")
    fys = nest(lambda s: f"{s[4]:.4f}")
    x = f"max(0,min(iw-iw/zoom,({fxs})*iw-iw/zoom/2))"
    y = f"max(0,min(ih-ih/zoom,({fys})*ih-{F['face_y']}*ih/zoom))"
    g = F["grade"]
    return (f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={W}x{H}:fps={fps},"
            f"eq=contrast={g['contrast']}:saturation={g['saturation']}:gamma={g['gamma']},"
            f"vignette=angle={g['vignette']}")


def loudnorm(src, dst, target):
    """Two-pass EBU R128 normalisation to the target loudness / true peak (linear gain when possible)."""
    I, TP = target["target_lufs"], target["true_peak_max"]
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(src), "-af",
                        f"loudnorm=I={I}:TP={TP}:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
    af = (f"loudnorm=I={I}:TP={TP}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", af, "-ar", "48000", "-ac", "2", str(dst)], check=True)
    return m


def measure_lufs(path):
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    return float(re.search(r"I:\s+(-?[\d.]+)", tail).group(1)), float(re.search(r"Peak:\s+(-?[\d.]+)", tail).group(1))


def assemble(ep, facecam=None, music=None, out=None, raw=None):
    ep = Path(ep)
    tokens = read_json(ROOT / "style/tokens.json")
    sc = read_json(ep / "scenes.json")
    raw = Path(raw or ep / "video_raw.mp4")
    narration = next(ep.glob("narration.*"))
    name = ep.name
    out = Path(out or ROOT / "output" / f"{name}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = ep / "assemble"
    tmp.mkdir(exist_ok=True)
    W, H, fps = sc["width"], sc["height"], sc["fps"]
    off0 = sc.get("offset", 0)

    # ---- audio ----
    nar = tmp / "narration_norm.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(off0), "-t", str(sc["duration"]), "-i", str(narration),
                    str(tmp / "narration_cut.wav")], check=True)
    loudnorm(tmp / "narration_cut.wav", nar, tokens["audio"])
    audio_in, audio_filter, amap = ["-i", str(nar)], None, None

    # ---- face-cam placement ----
    cam_shots = [s for s in sc["shots"] if s["kind"] == "facecam"]
    report = {"facecam": None}
    inputs = ["-i", str(raw)]
    vf = "[0:v]null[base]"
    if facecam and cam_shots:
        off, conf = clip_offset(facecam, narration)
        off -= off0
        report["facecam"] = {"clip": str(facecam), "offset_s": round(off, 2), "match": round(conf, 3)}
        print(f"[assemble] face-cam clip sits at narration t = clip t + {off:.2f} s (match {conf:.2f})")
        if conf < 0.25:
            print("[assemble] WARNING: face-cam audio barely matches the narration - check the clip/placement")
        inputs += ["-i", str(facecam)]
        parts, prev = [], "base"
        rd = ep / "render" / "out"
        # zoom on a slightly larger frame than the output (smoother push, less upscaling blur); 1.25x keeps it quick
        ss = tokens["facecam"].get("supersample", 1.25)
        ss_w, ss_h = int(round(W * ss / 2) * 2), int(round(H * ss / 2) * 2)
        for i, s in enumerate(cam_shots):
            cs, ce = s["start"] - off, s["end"] - off  # clip-time window for this shot
            # the clip may begin a moment after the shot (the planner cuts on the narration's sentence, the clip starts
            # when the camera rolled): keep lip-sync by placing the clip at its true time and holding its first frame
            # over the gap, instead of sliding the clip earlier
            gap = max(0.0, -cs)
            lbl = f"c{i}"
            zf = zoom_filter(s, facecam, off, W, H, fps, tokens["facecam"])
            vf += (f";[1:v]trim=start={max(0, cs):.3f}:end={ce:.3f},setpts=PTS-STARTPTS,"
                   f"tpad=start_duration={gap:.3f}:start_mode=clone,fps={fps},"
                   f"scale={ss_w}:{ss_h}:force_original_aspect_ratio=increase,crop={ss_w}:{ss_h},setsar=1,"
                   f"{zf},setpts=PTS-STARTPTS+{s['start'] - off0:.3f}/TB[{lbl}]")
            ov = rd / f"overlay_{s['id']}"
            if ov.exists() and any(ov.glob("f*.png")):  # popups / buttons drawn by the renderer, transparent
                k = inputs.count("-i")  # index of the input added next
                inputs += ["-framerate", str(fps), "-start_number", "0", "-i", str(ov / "f%04d.png")]
                vf += (f";[{k}:v]format=rgba,setpts=PTS-STARTPTS+{s['start'] - off0:.3f}/TB[o{i}];"
                       f"[{lbl}][o{i}]overlay=eof_action=pass:format=auto[{lbl}x]")
                lbl += "x"
            vf += (f";[{prev}][{lbl}]overlay=enable='between(t,{s['start'] - off0:.3f},{s['end'] - off0:.3f})'"
                   f":eof_action=pass[v{i}]")
            prev = f"v{i}"
        vf += f";[{prev}]null[vout]"
    else:
        vf += ";[base]null[vout]"

    # ---- optional music bed, ducked under the voice ----
    a_idx = inputs.count("-i")
    inputs += audio_in
    if music:
        mb = tokens["audio"]["music_bed"]
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        audio_filter = (f"[{a_idx + 1}:a]volume=-{mb['level_below_voice_db']}dB,aformat=channel_layouts=stereo[m];"
                        f"[m][{a_idx}:a]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=300[md];"
                        f"[{a_idx}:a][md]amix=inputs=2:duration=first:normalize=0[aout]")
        amap = "[aout]"
    fc = vf + (";" + audio_filter if audio_filter else "")
    cmd = ["ffmpeg", "-v", "error", "-stats", "-y", *inputs, "-filter_complex", fc, "-map", "[vout]",
           "-map", amap or f"{a_idx}:a", "-t", str(sc["duration"]),
           "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", "18", "-maxrate", "20M", "-bufsize", "40M",
           "-pix_fmt", "yuv420p", "-r", str(fps), "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)

    lufs, peak = measure_lufs(out)
    report.update({"output": str(out), "lufs": lufs, "true_peak_dbfs": peak})
    print(f"[assemble] {out}: {lufs} LUFS, peak {peak} dBFS")
    shutil.copyfile(ep / "scenes.json", out.with_suffix(".scenes.json"))
    write_json(ep / "assemble_report.json", report)
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--facecam")
    ap.add_argument("--music")
    ap.add_argument("--raw")
    ap.add_argument("--out")
    a = ap.parse_args()
    assemble(a.episode, a.facecam, a.music, a.out, a.raw)


if __name__ == "__main__":
    main()
