"""Mapani: narration audio -> finished 9:16 explainer reel.

    python make_reel.py --audio inputs/ep12            # one folder: audio.mp3 + news.txt + facecam.mp4 (optional)
    python make_reel.py --audio inputs/audio/ep12.mp3 [--episode ep12] [--out output/ep12.mp4]
    python make_reel.py --audio ... --mode map               # the older map-only format (one long map + face-cam)
    python make_reel.py --episode ep12 --from render        # re-run from a stage after editing work/ep12/scenes.json
    python make_reel.py --episode ep12 --only geocode       # run one stage

Each stage reads the previous stage's file in work/<episode>/ and writes its own, so any stage can be inspected,
hand-edited and re-run alone.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

from pipeline.common import ROOT, read_json, write_json

STAGES = ["transcribe", "entities", "geocode", "plan", "render", "review", "captions", "mix", "export"]
# story mode (default, the @adarshh.gupta format): Gemini storyboards the beats, pictures are made/fetched, and the
# review runs on the plan before the (slow) render
STORY_STAGES = ["transcribe", "entities", "geocode", "storyboard", "visuals", "plan", "review", "render", "captions",
                "mix", "export"]


def stage_storyboard(ep, a):
    from pipeline.storyboard import build
    clip, cam = find_facecam(ep, a), None
    if clip:  # where the face really is on screen (an edited source reel shows it only in parts)
        from pipeline.assemble import clip_offset
        from pipeline.facecam import windows
        cache = ep / "facecam_windows.json"
        key = [str(clip), clip.stat().st_mtime_ns]
        if cache.exists() and read_json(cache).get("key") == key:
            cam = read_json(cache)["windows"]
        else:
            off = clip_offset(clip, next(ep.glob("narration.*")))[0]
            cam = windows(clip, off)
            write_json(cache, {"key": key, "offset": off, "windows": cam})
    res = build(ep, a.story_model, facecam=cam)
    write_json(ep / "storyboard.json", res)
    kinds = {}
    for b in res["beats"]:
        kinds[b["visual"]] = kinds.get(b["visual"], 0) + 1
    return f"{len(res['beats'])} beats {kinds}"


def stage_visuals(ep, a):
    from pipeline.visuals import build
    res = build(ep, find_news=a.find_news, workers=a.workers)
    write_json(ep / "visuals.json", res)
    kinds = {}
    for v in res.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    return f"{kinds}"


def stage_transcribe(ep, a):
    from pipeline.transcribe import Transcriber
    src = Path(a.audio) if a.audio else next(ep.glob("narration.*"))
    if a.audio:  # keep a copy next to the work files so later re-runs don't depend on inputs/
        dst = ep / ("narration" + src.suffix.lower())
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    t = Transcriber()(src, a.lang)
    write_json(ep / "transcript.json", t)
    return f"{t['quality']['words']} words, repaired {t['repaired_spans'] or 'nothing'}"


def stage_entities(ep, a):
    from pipeline.entities import extract
    res = extract(read_json(ep / "transcript.json"), a.model, cache_dir=ep / "cache")
    write_json(ep / "entities.json", res)
    kinds = {}
    for e in res["entities"]:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    return f"{kinds}, {len(res['dropped'])} dropped"


def stage_geocode(ep, a):
    from pipeline.geocode import geocode
    res = geocode(read_json(ep / "entities.json"), offline=a.offline)
    write_json(ep / "geo.json", res)
    return f"{len(res['items'])} resolved, {len(res['missing'])} missing, {len(res['needs_review'])} to review"


def stage_plan(ep, a):
    if a.mode == "story":
        from pipeline.plan import validate
        from pipeline.story_plan import StoryPlanner, brand_handle
        sc = StoryPlanner(ep, read_json(ROOT / "style/tokens.json"), a.handle or brand_handle(ep)).plan()
        validate(sc)
        write_json(ep / "scenes.json", sc)
        kinds = {}
        for s in sc["shots"]:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
        return f"{len(sc['shots'])} shots {kinds}"
    from pipeline.plan import Planner, validate
    sc = Planner(ep, read_json(ROOT / "style/tokens.json")).plan()
    validate(sc)
    write_json(ep / "scenes.json", sc)
    from pipeline.pictures import add as add_pictures  # needs the written plan: cards go where no text is
    n_pics = 0 if a.offline else add_pictures(ep)  # map mode only
    return f"{len(sc['shots'])} shots, {sum(len(s.get('layers', [])) for s in sc['shots'])} layers, {n_pics} pictures"


def stage_render(ep, a):
    from pipeline.render_stage import render
    return f"{render(ep, workers=a.workers)}"


def stage_review(ep, a):
    from pipeline.render_stage import render
    from pipeline.review import review
    if a.no_review:
        return "skipped (--no-review)"
    # story mode reviews the plan before rendering: its fixes need no re-render
    rep = review(ep, fix=True, rerender=None if a.mode == "story" else lambda: render(ep, workers=a.workers))
    return (f"{len(rep['fixed'])} fixed, {len(rep['fails'])} problems left, {len(rep['warnings'])} warnings "
            f"-> work/{ep.name}/review.md")


def stage_captions(ep, a):
    tok = read_json(ROOT / "style/tokens.json")["captions"]
    # the reference reels have no running subtitles; keyword/number/place text is drawn by the render stage
    return "word-by-word captions off (style: keyword popups only)" if not tok["word_by_word"] else \
        "word-by-word captions are on in tokens.json, but not built yet"


def find_facecam(ep, a):
    if a.facecam:
        return Path(a.facecam)
    vids = (".mp4", ".mov", ".mkv")
    # one folder per episode (inputs/<ep>/facecam.mp4), or the older inputs/audio/<ep>_facecam.mp4
    d = ROOT / "inputs" / ep.name
    cams = sorted(d.glob("*facecam*.*")) + sorted(d.glob("*cam*.*")) if d.is_dir() else []
    hit = next((p for p in cams if p.suffix.lower() in vids), None)
    if hit:
        return hit
    for d in (ROOT / "inputs" / "audio", ROOT / "inputs" / "facecam"):
        hit = next((p for p in sorted(d.glob(f"{ep.name}_facecam.*")) if p.suffix.lower() in vids), None)
        if hit:
            return hit
    return None


AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")


def episode_folder(path):
    """--audio inputs/ep13 (a folder): the narration inside it (audio.* / narration.* / the only audio file), and the
    folder's name as the episode. Its news.txt and facecam.* are picked up by the later stages."""
    d = Path(path)
    files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in AUDIO_EXT]
    pick = next((p for p in files if p.stem.lower() in ("audio", "narration", "voice", "vo")), files[0] if files else None)
    if pick is None:
        raise SystemExit(f"no audio file ({', '.join(AUDIO_EXT)}) in {d}")
    return pick, d.name


def stage_mix(ep, a):
    from pipeline.assemble import assemble
    out, rep = assemble(ep, find_facecam(ep, a), a.music, a.out)
    cam = rep["facecam"]
    return (f"{out}  {rep['lufs']} LUFS / peak {rep['true_peak_dbfs']} dBFS; face-cam "
            + (f"placed (offset {cam['offset_s']} s, match {cam['match']})" if cam else "placeholder (no clip given)"))


def stage_export(ep, a):
    from pipeline.style_capture.contact_sheet import sheet
    out = Path(a.out or ROOT / "output" / f"{ep.name}.mp4")
    sc = read_json(ep / "scenes.json")
    segs = [{"start": s["start"], "end": s["end"], "kind": s["kind"]} for s in sc["shots"]]
    words = read_json(ep / "transcript.json")["words"]
    sheet(out, segs, words, out.with_suffix(".sheet.jpg"), title=f"{ep.name}  (bar: blue=map, red=face-cam)")
    return f"{out}, {out.with_suffix('.scenes.json').name}, {out.with_suffix('.sheet.jpg').name}"


RUN = {"transcribe": stage_transcribe, "entities": stage_entities, "geocode": stage_geocode, "plan": stage_plan,
       "storyboard": stage_storyboard, "visuals": stage_visuals,
       "render": stage_render, "review": stage_review, "captions": stage_captions, "mix": stage_mix, "export": stage_export}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", help="narration file (mp3/wav/m4a)")
    ap.add_argument("--episode", help="episode name; default = audio file name")
    ap.add_argument("--out", help="final mp4 (default output/<episode>.mp4)")
    ap.add_argument("--mode", choices=["story", "map"], default="story", help="story (default) or the map-only format")
    ap.add_argument("--from", dest="start", choices=STORY_STAGES, help="start at this stage")
    ap.add_argument("--only", choices=STORY_STAGES, help="run just this stage")
    ap.add_argument("--to", choices=STORY_STAGES, default="export", help="stop after this stage")
    ap.add_argument("--story-model", default="gemini-3.1-pro-preview", help="Gemini model for the storyboard")
    ap.add_argument("--find-news", action="store_true", help="search the web for news clippings (else inputs/<ep>/news.txt)")
    ap.add_argument("--handle", help="creator handle for watermark + end card (default inputs/brand/brand.json)")
    ap.add_argument("--lang", help="force transcription language, e.g. hi")
    ap.add_argument("--model", default="gemini-3.8-flash", help="Gemini model for entities")
    ap.add_argument("--offline", action="store_true", help="geocode without OpenStreetMap lookups")
    ap.add_argument("--facecam", help="face-cam clip (default: inputs/audio/<episode>_facecam.mp4 if present)")
    ap.add_argument("--music", help="optional music bed (only files you have the rights to)")
    ap.add_argument("--workers", type=int, default=3, help="parallel render workers (3 keeps a laptop cool)")
    ap.add_argument("--no-review", action="store_true", help="skip the automatic review/fix loop")
    a = ap.parse_args()
    if not a.audio and not a.episode:
        ap.error("give --audio (new episode) or --episode (existing work folder)")
    if a.audio and Path(a.audio).is_dir():
        audio, folder_name = episode_folder(a.audio)
        a.audio = str(audio)
        name = a.episode or folder_name
    else:
        name = a.episode or Path(a.audio).stem
    ep = ROOT / "work" / name
    ep.mkdir(parents=True, exist_ok=True)
    order = STORY_STAGES if a.mode == "story" else STAGES
    for st in (a.start, a.only, a.to):
        if st and st not in order:
            ap.error(f"stage {st} is not part of --mode {a.mode}")
    if a.only:
        todo = [a.only]
    else:
        first = order.index(a.start) if a.start else (0 if a.audio else None)
        if first is None:
            ap.error("--episode without --audio needs --from or --only")
        todo = order[first:order.index(a.to) + 1]
    for s in todo:
        t0 = time.time()
        print(f"== {s}")
        msg = RUN[s](ep, a)
        print(f"   {msg}  ({time.time() - t0:.1f}s)")
    print(f"done: work/{name}/")


if __name__ == "__main__":
    sys.exit(main())
