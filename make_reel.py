"""Mapani: narration audio -> finished 9:16 map-explainer reel.

    python make_reel.py --audio inputs/audio/ep12.mp3 [--episode ep12] [--out output/ep12.mp4]
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

STAGES = ["transcribe", "entities", "geocode", "plan", "render", "captions", "mix", "export"]


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


def not_built(name):
    def run(ep, a):
        raise SystemExit(f"stage '{name}' is not built yet (build step 4-6 in REEL_PIPELINE.md)")
    return run


RUN = {"transcribe": stage_transcribe, "entities": stage_entities, "geocode": stage_geocode,
       **{s: not_built(s) for s in STAGES[3:]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", help="narration file (mp3/wav/m4a)")
    ap.add_argument("--episode", help="episode name; default = audio file name")
    ap.add_argument("--out", help="final mp4 (default output/<episode>.mp4)")
    ap.add_argument("--from", dest="start", choices=STAGES, help="start at this stage")
    ap.add_argument("--only", choices=STAGES, help="run just this stage")
    ap.add_argument("--to", choices=STAGES, default="export", help="stop after this stage")
    ap.add_argument("--lang", help="force transcription language, e.g. hi")
    ap.add_argument("--model", default="gemini-3.8-flash", help="Gemini model for entities")
    ap.add_argument("--offline", action="store_true", help="geocode without OpenStreetMap lookups")
    a = ap.parse_args()
    if not a.audio and not a.episode:
        ap.error("give --audio (new episode) or --episode (existing work folder)")
    name = a.episode or Path(a.audio).stem
    ep = ROOT / "work" / name
    ep.mkdir(parents=True, exist_ok=True)
    if a.only:
        todo = [a.only]
    else:
        first = STAGES.index(a.start) if a.start else (0 if a.audio else None)
        if first is None:
            ap.error("--episode without --audio needs --from or --only")
        todo = STAGES[first:STAGES.index(a.to) + 1]
    for s in todo:
        t0 = time.time()
        print(f"== {s}")
        msg = RUN[s](ep, a)
        print(f"   {msg}  ({time.time() - t0:.1f}s)")
    print(f"done: work/{name}/")


if __name__ == "__main__":
    sys.exit(main())
