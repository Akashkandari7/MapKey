# ReelMimic: what we reuse (build step 1)

Cloned to `vendor/reelmimic` (MIT, shallow clone, 2026-10-18). Nothing in `vendor/` is edited; anything we take is
copied into `pipeline/` with its license header kept.

## Decisions so far

| Topic | Decision |
|---|---|
| Entity LLM (stage 2) | **Gemini** (key via `GEMINI_API_KEY` env var, never stored in the repo) |
| India boundary | **DataMeet** SoI-aligned outline (github.com/datameet/maps), license logged in `data/geo/SOURCES.md` |
| Creator on camera | No, maps only |
| Reference reels | You drop them into `inputs/reference_reels/` |

## Reuse as-is (copied into our tree)

| ReelMimic piece | Path in vendor | What we use it for |
|---|---|---|
| `analyze.py` | `.claude/skills/video-clone/scripts/` | Phase 1. Scene detection, 1 fps + per-shot contact sheets, shot lengths, camera-move stats, audio loudness/beat report. We wrap it and add color sampling, caption cropping, and map-color clustering. |
| `compare.py` | same | Review loop. Reference vs. our output side by side, with brightness/contrast/colorfulness/camera-pace numbers per shot. |
| `motion_check.py`, `clip_strip.py` | same | Strips of every frame across a zoom, to check easing and that motion lands on the word. |
| engine-kit `kit.js` + `render.mjs` | `.claude/skills/video-clone/assets/engine-kit/` | **The render core** (see below). Frame = pure f(t), parallel resumable frame rendering in headless Chrome, contact sheets, crops, encode. |
| Style-file format | `.claude/skills/video-clone/styles/_TEMPLATE.md` | Shape of `styles/map-explainer-reel.md` (frontmatter + recognition cues / production defaults / known pitfalls). |

## Reuse as knowledge (rules we follow, code we rewrite)

From `motion-graphics/categories/maps/module.md`, lessons that bit them:
- Animate the camera continuously and keep it eased (easeInOutCubic); no decorative glows or particles unless the reference has them.
- Filter overseas territories with a lon/lat keep-box around the mainland (France ≠ French Guiana).
- Unwrap longitudes around the camera center to handle the antimeridian (Russia, Fiji).
- Labels must be offset from shapes and from each other, clamped to the safe area, and readable for at least 0.3 s.
- Any third-party basemap imagery needs an on-screen credit.

From `hyperframes-core/references/determinism-rules.md`: no clocks, no unseeded random, no network at render time, no state carried between frames.

## Not reused, and why

| Piece | Why not |
|---|---|
| **HyperFrames DOM/GSAP compositions** | Its map lane bakes the camera move into a video, then lays SVG overlays on top **only while the camera holds still**, because a moving camera and fixed projected paths drift apart. Our reels pan and zoom *while* highlights, routes and labels are on screen. So we draw the whole map (land, borders, highlights, routes, labels) on canvas each frame from one camera = f(t). Everything stays locked to the geography. |
| `bake-basemap.mjs` (MapLibre + Esri/CARTO tiles) | Needs live third-party tiles (ToS and network), and its `world-atlas` country data uses non-SoI India borders. We render from offline GeoJSON instead. If the reference uses satellite or terrain imagery, I'll raise it in Phase 1. |
| `world-atlas` / raw Natural Earth for India | Shows disputed borders. India comes from DataMeet; Pakistan and China polygons get clipped so nothing draws over the SoI outline. |
| ReelMimic web app, director/builder agent flow | Built for 1–3 hour planned projects. We need one command driven by the audio. |
| Their transcriber (Parakeet/whisper.cpp) | We use faster-whisper large-v3 (CUDA, RTX 4050), which handles Hindi/Hinglish with word timestamps. |
| Bundled SFX (Pixabay) | Listed in the manifest but the `.mp3`s aren't in the repo (downloaded on demand). We use what you put in `inputs/assets/`, or Pixabay-licensed files logged in `ASSETS_LICENSES.md`. |

## Render engine plan (justification for the switch)

The spec asks for HyperFrames "unless a clearly better option exists". I'm using **ReelMimic's own engine-kit** (its
second engine, the one its painted/pixel/whiteboard styles run on) rather than the HyperFrames DOM engine:

- **Camera-locked overlays:** canvas + `d3-geo` projection evaluated per frame, so zooms and pans never drift from borders.
- **One file per scene:** `render/scenes/<id>.js`, each calling a shared `map.js` / `captions.js` / `style.js`. All style values come from `tokens.json`.
- **Deterministic and parallel:** `render.mjs --frames --workers=N` already exists, with resumable JPEG frames.
- **Speed budget:** 60 s × 30 fps = 1800 frames in under 15 min means about 0.5 s per frame per worker with 1 worker, or about 3.5 s with 8. Canvas 2D with pre-simplified GeoJSON should land around 50–150 ms/frame.

Changes to the kit: `W/H` becomes 1080×1920 and is read from tokens. Scenes get generated from `scenes.json` instead of being written by hand.

## Your machine (checked)

git 2.55 · Python 3.13 · ffmpeg 9.0.2 · Node 24.19 · Chrome (installed) · RTX 4050 6 GB · 16 threads.
