# MapKey (Mapani): narration audio → finished map-explainer reel

Give it a voiceover (Hindi / Hinglish / English) and it returns a 1080×1920 Instagram reel in the style of the
reference page: satellite map, the place being talked about lifted off the map, metro lines and roads drawing on,
labels and stats typing in on the spoken word, and your face-cam outro spliced in with punch-in zooms.

```
python make_reel.py --audio inputs/audio/ep12.mp3
```

The full spec is in [REEL_PIPELINE.md](REEL_PIPELINE.md); the measured style in [style/STYLE.md](style/STYLE.md).

---

## 1. Setup (once)

Needs: Windows/macOS/Linux, **Python 3.13**, **Node 20+**, **ffmpeg** on PATH, **Google Chrome**, an NVIDIA GPU
for fast transcription (CPU works, slower).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # macOS/Linux: .venv/bin/python
cd pipeline/render && npm install && cd ../..
.venv/Scripts/python -m pipeline.geo.fetch_raw               # ~60 MB of open map data, then builds the geo database
```

**Gemini key** (used once per reel to pick out places, lines and numbers; about ₹2 per reel):
copy `.env.example` to `.env` and put your key after `GEMINI_API_KEY=`. Keys from Google AI Studio / Cloud
Console now start with `AQ.` (bound to a service account). If a fresh key says 401, wait a few minutes and retry;
make sure nothing extra was pasted before or after the key. `.env` is never committed.

## 2. Make a reel

Put files in `inputs/` (never modified, never committed):

```
inputs/audio/ep12.mp3             the narration
inputs/audio/ep12_facecam.mp4     (optional) your on-camera clip, found automatically by name
inputs/assets/                    (optional) music you have the rights to, etc.
```

```
python make_reel.py --audio inputs/audio/ep12.mp3
```

Output in `output/`:
- `ep12.mp4`: the reel (1080×1920, 30 fps, H.264, −14 LUFS)
- `ep12.sheet.jpg`: one frame per second with the words spoken, for a quick look
- `ep12.scenes.json`: the shot list that was rendered

and `work/ep12/review.md`: what the automatic review checked, fixed, and wants you to look at.

Time on a laptop (RTX 4050): about 5 min for a 50 s reel in an area seen before; 12–15 min the first time in a
new area (satellite imagery + OpenStreetMap lookups are downloaded once and cached).

Options: `--facecam path`, `--music path`, `--workers 3` (parallel render tabs; 3 keeps a laptop cool),
`--no-review`, `--offline` (no OpenStreetMap lookups).

## 3. Change something by hand (manual override)

Every stage writes its result to `work/<episode>/`, and any stage can be re-run alone:

| Stage | Writes | What it does |
|---|---|---|
| transcribe | `transcript.json` | faster-whisper large-v3 (+ large-v2 repair of dropped speech), word timestamps |
| entities | `entities.json` | Gemini: places, lines, numbers, key facts, sections (hook / explain / opinion / CTA) |
| geocode | `geo.json` | real geometry: offline Survey-of-India-correct database, then OpenStreetMap |
| plan | `scenes.json` | camera moves, lines, labels, stats, face-cam punch-ins, all timed to the words |
| render | `video_raw.mp4` | headless Chrome draws every frame (satellite map, overlays) |
| review | `review.md` | checks and fixes (below), re-renders if it fixed something |
| captions | – | off by style: the page uses keyword popups, not running subtitles |
| mix | `output/*.mp4` | face-cam spliced by its audio, grade + zooms, loudness to −14 LUFS |
| export | sheet + scenes | contact sheet and a copy of scenes.json next to the video |

Edit `work/ep12/scenes.json` (move a label, change a zoom, delete a stat, retime something), then:

```
python make_reel.py --episode ep12 --from render
```

Add `"locked": true` to any layer you edited so the review never moves it back. Useful fields:
- camera keys: `t0` (move starts), `t` (move ends), `x`, `y` (Web Mercator metres), `mpp` (metres per pixel, smaller = closer)
- `route`: `in` (starts drawing), `draw_s`, `color`; `pin`: `label`, `side` (above/below/left/right), `in`, `out`
- `stat`: `lines[].text`, `lines[].in`, `out`; text on the map has `X`, `Y` (where it sits on the map)
- face-cam shot: `punches` (`t` seconds into the shot, `scale`), `layers` (popups, Follow/COMMENT buttons)

Other re-runs: `--only geocode`, `--from plan`, `--from mix --facecam clip.mp4`.

## 4. The automatic review

Runs after render, before the final mix. It computes every label and stat on screen every 0.25 s and checks:

- every place/line spoken while the map is on screen is shown (adds a missing label if it can)
- no text outside the Instagram safe area (moves it in; labels switch side instead)
- no stat or title covering a line or a label; no labels overlapping (moves them clear)
- every graphic lands on its word within ±0.3 s (snaps it back)
- pacing, camera moves and map colour inside the reference's measured range (warnings)
- places that came from a guess or from an announced station list are listed for you to check

Fixed problems trigger a re-render and a second check (max 2 rounds). Read `work/<ep>/review.md` before posting.

## 5. Accuracy rules (built in)

- **India's borders follow the Survey of India** (all of J&K and Ladakh incl. Aksai Chin, Gilgit-Baltistan,
  Shaksgam); no neighbouring country is ever drawn over it. See `data/geo/SOURCES.md`.
- **No invented lines.** A line is drawn from OpenStreetMap, or (for a planned line OSM doesn't have) through its
  announced stations' real locations, checked for consistency. If neither is reliable, the line is not drawn and
  only its places are labelled. Station-based lines are always flagged in the review.
- Labels say places the way the narrator said them; numbers appear on the number word.
- **Counted groups.** "23 provinces" fills every admin-1 unit of that type; "4 municipalities: Beijing, Shanghai,
  ..." fills exactly the places named after the count, each on its word, in one colour per group. Land the country
  claims but doesn't hold (Taiwan for China) gets a dashed outline, never a fill. If the total is said again at the
  end, a recap lists every group in its colour.

## 6. Change the style

All style values live in [style/tokens.json](style/tokens.json) (colours, fonts, sizes, timings, camera feel,
face-cam punch-ins, grade). The renderer reads nothing else. Three looks were measured from the reference reels:
`metro_dark` (default), `bright_political`, `hillshade_glow`; set `default_look` to switch.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `GEMINI_API_KEY is not set` / 401 | see Setup; the key must be exactly what Google shows, nothing before it |
| transcription very slow | CUDA not found: check `nvidia-smi`; the pipeline falls back to CPU automatically |
| render tab never becomes ready | update Chrome; `node pipeline/render/probe_ready.mjs 3` (from `work/<ep>/render`) shows page errors |
| a place is in the wrong spot | `work/<ep>/geo.json` shows how it was found; fix the entity in `entities.json` and re-run `--from geocode` |
| Overpass errors (502/504) | OpenStreetMap servers are busy; re-run, results are cached once fetched |

## 8. Credits and licences

Map data: © OpenStreetMap contributors (ODbL); DataMeet India boundaries (CC0 / CC BY 4.0); Natural Earth (public
domain); GeoNames (CC BY 4.0). Imagery: contains modified Copernicus Sentinel data; NASA Blue Marble (public domain). Fonts: Inter, Poppins, Playfair
Display (SIL OFL). Render runtime adapted from [ReelMimic](https://github.com/edenfunf/reelmimic) (MIT). Details:
[ASSETS_LICENSES.md](ASSETS_LICENSES.md), [data/geo/SOURCES.md](data/geo/SOURCES.md). The on-screen credit line is
added automatically.
