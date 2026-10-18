# Map-Explainer Reel Pipeline: Build Spec

> Hand this file to Claude Code at the root of a new project folder.
> Goal: **I give a narration audio file → pipeline returns a finished 9:16 reel in the exact style of the reference page.**

---

## 1. Context

- I edit Instagram reels for the page **@creator.adarshh**. The format: **map animations** (zooms, pans, highlighted countries/regions, routes, labels) synced to a **voiceover explanation**.
- I want this editing done by AI. My only input per video should be the **narration audio** (Hindi / Hinglish / English).
- Base repo to study and reuse: **https://github.com/edenfunf/reelmimic** (MIT). It already has:
  - a style system (`.claude/skills/video-clone/styles/<name>.md`)
  - a motion-graphics engine (HyperFrames) with frame preview, contact sheets and MP4 export
  - a reference-video breakdown (shot lengths, transitions, colors, camera moves)
- ReelMimic alone is NOT enough because: it makes "similar" not "exact" output, each video is a 1–3 hour planned project, it draws maps freehand (not real geo data), and it isn't driven by my audio track. So we **reuse its pieces** and build a fast, repeatable, audio-driven pipeline.

---

## 2. Inputs I will provide

```
inputs/
  reference_reels/     3–5 MP4s of the page's reels (style source of truth)
  assets/              (optional) fonts, logo/watermark, map base, SFX, music, color codes, any editing template
  audio/               narration file for each new video (mp3/wav/m4a)
```

If anything needed is missing, **ask me** instead of guessing.

---

## 3. Phase 1: Style capture (do this first, stop and show me)

Analyze every reference reel and write **`STYLE.md`**, a measurable style spec. Use ffmpeg scene detection, frame extraction (every 0.5 s + every cut), and color sampling. Cover:

| Area | What to capture |
|---|---|
| Format | resolution, fps, length range, safe areas |
| Map look | projection, land/water/border colors (hex), border thickness, labels (font, size, case), terrain/flat, shadows/glow, background |
| Highlights | how a country/region/city is highlighted (fill, outline, pulse, glow), colors |
| Camera | zoom types, pan speed, easing curves, typical zoom durations, how it moves between locations |
| Overlays | arrows, route lines (animated draw?), markers, icons, flags, images, year/date tags, number counters |
| Captions | font, weight, size, color, stroke/shadow, position, words per chunk, word highlight style, entry animation |
| Editing rhythm | average shot length, cuts relative to speech (on sentence? on keyword?), intro hook, outro/CTA |
| Transitions | types and durations |
| Sound | background music level vs voice, SFX (whoosh on zoom, pop on highlight), ducking |
| Branding | logo/watermark, position, opacity |

Also produce:
- `style/contact_sheets/`: frame grids per reference reel, annotated
- `style/tokens.json`: all colors, fonts, sizes, durations, easings as machine-readable tokens (the renderer must read ONLY from this, no hardcoded style values)
- `styles/map-explainer-reel.md`: same style in ReelMimic's style-file format (frontmatter + Recognition cues / Production defaults / Known pitfalls)

**Checkpoint:** show me STYLE.md + contact sheets. Don't start Phase 2 until I approve.

---

## 4. Phase 2: The pipeline

### One command
```
python make_reel.py --audio inputs/audio/ep12.mp3 --out output/ep12.mp4
```

### Stages (each stage writes its output to `work/<episode>/` so it can be inspected and re-run alone)

1. **Transcribe** → `transcript.json`
   - faster-whisper (large-v3) with **word-level timestamps**; must handle Hindi/Hinglish. Optionally output Devanagari + Roman versions; captions follow whatever script the reference reels use.
2. **Understand** → `entities.json`
   - LLM pass over transcript: places (country/state/city/river/region), routes ("from A to B"), dates/years, numbers, key phrases to emphasize. Each with start/end time.
3. **Geocode** → `geo.json`
   - Resolve every place to geometry from **offline real map data** (Natural Earth / geoBoundaries GeoJSON; Nominatim only as fallback, cached).
   - **India boundaries must follow the official Survey of India map** (legal requirement for Indian publishing). Never use a disputed-border dataset for India.
4. **Plan scenes** → `scenes.json`
   - Turn entities + timing into a shot list: camera target, zoom level, highlights, overlays, captions, SFX cues. Rules come from STYLE.md (e.g. "camera starts moving 0.3 s before the place name is spoken").
   - Validate against a JSON schema. Every shot fully covers the audio, no gaps.
5. **Render** → `video_raw.mp4`
   - Use ReelMimic's HyperFrames engine (or the engine-kit) unless a clearly better option exists; justify it if you switch (e.g. Remotion).
   - Rules: one file per scene, shared map/caption/style files, **deterministic frames** (frame = f(time), no randomness), parallel rendering.
6. **Captions** → burned in, word-synced from `transcript.json`, styled from `tokens.json`.
7. **Audio mix** → narration + background music + SFX, with ducking, loudness normalized (-14 LUFS for Instagram).
8. **Export** → `output/<episode>.mp4`, 1080×1920, H.264, Instagram-ready. Also export `scenes.json` + a contact sheet so I can review quickly.

### Review loop (automatic, before export)
- Render a contact sheet of the output and compare against the reference contact sheets: colors, caption style, zoom feel, pacing.
- Check: every spoken place appears on screen, no wrong/missing borders, captions never cover the highlighted area, no text outside safe area.
- Fix and re-render failures. Report what was fixed.

### Manual override
I should be able to edit `scenes.json` by hand (change a zoom, add a highlight) and re-run only from stage 5:
```
python make_reel.py --episode ep12 --from render
```

---

## 5. Project structure

```
inputs/            my files (never modify)
style/             STYLE.md, tokens.json, contact sheets
styles/            ReelMimic-format style file
pipeline/          transcribe.py, entities.py, geocode.py, plan.py, render/, captions.py, mix.py
data/geo/          cached GeoJSON (Natural Earth, geoBoundaries, Survey of India boundary)
work/<episode>/    intermediate outputs per stage
output/            final videos
make_reel.py       entry point
README.md          setup + usage
```

---

## 6. Build order (incremental, check in with me after each)

1. Clone & study ReelMimic: engine, style files, preview/export scripts. Write a short note on what we reuse.
2. Phase 1: style capture → **my approval**
3. Stages 1–3 (transcript, entities, geo) on one test audio → show me the JSONs
4. Stage 4–5: render a single 10-second scene in style → **my approval on look**
5. Full pipeline end-to-end on one audio → I compare with my manual edit
6. Review loop + manual override + README

---

## 7. Acceptance criteria

- Side-by-side with a real reel from the page, a viewer can't easily tell which is AI-made.
- Every place mentioned in the audio is shown and correctly located.
- Map motion lands on the spoken word (±0.3 s).
- 60-second reel renders in **< 15 minutes** on my machine.
- Zero manual editing needed for a normal episode; small fixes only via `scenes.json`.

---

## 8. Rules

- Style values come only from `tokens.json` / STYLE.md, never hardcoded.
- Log the source and license of every asset, font, music or dataset used.
- Don't download copyrighted music; use only what I put in `inputs/assets/` or properly licensed sources.
- No generated footage of real people. If the reference reels show the creator on camera, leave those segments as placeholders for me.
- When unsure about a style detail, ask me rather than guess.
