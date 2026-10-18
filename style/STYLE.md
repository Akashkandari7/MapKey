# @creator.adarshh map reels: measured style spec

Source: the 6 reels in `inputs/reference_reels/` (720×1280, 42–55 s). Everything here was measured with the scripts in
`pipeline/style_capture/` or checked frame by frame. Machine-readable values live in `style/tokens.json`; the
renderer reads only that file.

Evidence files: `style/contact_sheets/r1..r6.jpg` (1 frame/s, labelled), `style/frames/sync/*.jpg` (sync and
transition strips), `style/moves.json` (camera moves), `style/palette.json` (colours), `style/audio.json` (sound),
`style/transcripts/` (word timings), `style/segments.json` (map / b-roll / face-cam timeline).

---

## 0. The big finding: three looks, one format

The six reels share one **format** (editing, pacing, sync, structure) but use **three visual looks**, probably from
different editors or periods:

| Look | Reels | One-line description |
|---|---|---|
| **A · metro_dark** | r1 Rapid Metro, r2 NY Metro, r6 Pink Line (the three newest) | Dim, desaturated satellite; area outside the subject darkened; strong vignette; thick glowing route lines; yellow + neon-green stats; SF-Pro-style sans |
| **B · bright_political** | r3 Pangaea, r4 NH44 | Vivid satellite; flat, fully opaque saturated fills (countries/states); white boxes, red ribbons; Poppins |
| **C · hillshade_glow** | r5 Sikkim | Greyscale terrain relief; neon-green glowing region; spotlight + very strong vignette; bold italic serif headlines |

The pipeline supports all three as presets in `tokens.json` (`looks.*`). **Default = A**, because it is the newest and
the most common. ➜ *Needs your call: is A right as the default, and should B/C stay as options?*

---

## 1. Format

| Item | Value | How measured |
|---|---|---|
| Output | 1080×1920, 30 fps (refs are 720×1280 Instagram downloads, 29.97/30 fps) | ffprobe |
| Length | 42–55 s, typical 47 s | ffprobe |
| Structure | map story → face-cam outro (r1, r2, r5, r6: last 12–17 s); r3 intercuts face-cam all through; r4 map only | `segments.json` |
| Safe area | Titles at y = 13–22 % of height; CTA pill at 7–12 %; nothing important in the bottom 400 px or right 140 px (IG UI) | text bounding boxes |

## 2. Map look (per look)

| | A · metro_dark | B · bright_political | C · hillshade_glow |
|---|---|---|---|
| Basemap | Satellite, brightness 0.24–0.32, saturation 0.23–0.25 | Satellite, brightness 0.30–0.41, saturation 0.59–0.62 | Greyscale relief, brightness 0.18, saturation 0.07 |
| Outside subject | Darkened (city/borough boundary mask) | Not darkened | Spotlight falloff |
| Vignette (centre→corner) | 1.00 → 0.26–0.36 | 1.00 → 0.40–0.65 | 1.00 → 0.15 |
| Region highlight | Blue colour-tint over the imagery, texture visible (#0F688E–#1689B8), r2 | Flat opaque fills: India **#008BFE**; states/continents categorical; thin red outline + glow (r3) | Neon green #25B928–#33D03C, wide outer glow, texture visible |
| Borders | Only the subject boundary (soft edge of the dim mask) | Subject fills only; no internal border lines between states | Subject only |
| Labels on map | Yellow pill **#F6F015**, black text, pointer | White box, black text | Red sticker **#B30A0C**, white caps, pointer |
| Terrain/3D | Flat, top-down, no tilt | Flat, top-down | Relief shading only |

India in r4/r5 is drawn with the **full SoI outline** (all of J&K and Ladakh), consistent with the DataMeet dataset we chose.

## 3. Highlights, routes, markers

- **Routes (A):** 14–15 px wide at 1080, round caps. Colours measured on core pixels: red **#E50007**, yellow
  **#F6FF15**, magenta **#E60082**, pink **#EA52A9**. Slight glow and dark drop shadow. They **draw on** along the path
  in ~1.0–1.2 s (r1, r6), starting on the spoken line name.
- **Stations (A):** red dot (r ≈ 9 px) with a 3 px white ring; the label pill pops in (scale 0.6→1, ~0.3 s) 0.1–0.2 s
  after the station is named.
- **Highways (B):** thin dark dotted line, red circle end markers with white ring, white name boxes (Srinagar,
  Kanyakumari); route name as huge white glowing text rotated along the road.
- **Region reveal:** B = instant pop, one region per spoken name (r4 states); A/C = ~0.2 s fade.

## 4. Camera

Measured with ORB feature tracking (`moves.py`): 58 moves across the six reels.

| | A | B | C |
|---|---|---|---|
| Move length | 3.5–6.5 s | 1.6–3.0 s | 1.5–4.0 s |
| Zoom per move | ×1.2–2.0 | ×0.45–2.3 (zoom out to context, in to detail) | ×1.7–3.4 punch-ins |
| Easing | near-linear, soft ease-out (50 % of progress at 56–75 % of time) | easeInOutSine (8/13 moves) | easeOutCubic (50 % done in first 20–25 %) |
| Between moves | never fully still: slow drift zoom | holds | holds |
| Rotation | none | none (r4 text is rotated, not the camera) | none |

- **Timing vs speech:** move start − nearest word start = **median 0.02 s**, range −0.3…+0.25 s (37 moves). The camera
  starts moving *on* the place name, not before it. (The spec suggested 0.3 s early; the references say ~0.1 s early at
  most, so `lead_before_word = 0.1`.)
- **Between places:** zoom out to show both, then zoom in (r4 India → Tamil Nadu, r5 Sikkim → India → Sikkim). No
  flyover pans across long distances.

## 5. Overlays and text

| Element | Look A | Look B | Look C |
|---|---|---|---|
| Title | White small line + big yellow word ("New york / **Metro**"), or one big white line ("Pink line"); y ≈ 0.18–0.22, slowly grows | Big white glowing caps ("WORLD") + red ribbon | Bold italic serif, green/white/lime words |
| Stat | Label in yellow (#F4E41A, ~50 px), value in neon green **#36EE2E** heavy (~96 px), dark drop shadow, placed over the subject | White caps | Number + yellow/purple sticker ("PER CAPITAL INCOME") |
| Font (seen) | SF Pro Display/Text | Poppins | Playfair-like serif |
| Font (we use) | **Inter** (free; SF Pro's licence doesn't allow this use) | Poppins | Playfair Display |

- **Entry animation:** typewriter. Characters appear left→right over ~0.35 s, starting **0.0–0.2 s after the word
  starts** (sync strips: "Pink"→"Pink line", "11"→"11.7 KM", "25 Subway Ser…"). Pills/labels scale-pop. Exits fade.
- **Density:** a new visual event (label, stat, fill, route) every 1–2 s while the map is on screen.
- **Captions:** **no word-by-word subtitles** in any reel. Only keywords, numbers and place names. (Confirmed with you.)

## 6. Editing rhythm

- Voice is continuous: **no pause longer than 0.35 s** in any reel (dead air cut out). 3.2–3.5 words/s.
- All cuts are **hard cuts** (map→face-cam, map→b-roll). No transitions, wipes or flashes, except one orange flash
  at the start of r5.
- **Hook:** the subject is on screen from frame 0 with the camera already moving and a route drawing or region filling;
  the title types in by 0.2–0.5 s. No title card, no intro.
- **Outro:** creator on camera (12–17 s) with an opinion/question for the audience; or (r4) a "comment.." pill over the
  final map for ~3 s.
- **B-roll** shows exactly what is being said: full-frame stock video with the spoken stat on top ("24/7
  Operational", "Annual Ridership 1.28 Billion" over subway footage, r2), or stacked rounded photo cards with a soft glow
  (r5 farming).

## 7. Sound

| Item | Measured |
|---|---|
| Loudness | **−14.1…−14.3 LUFS integrated** on all six (the IG target); true peak −0.0…−2.8 dBTP (we cap at −1) |
| Dynamics | LRA 1.1–3.1 LU: heavily compressed voice |
| Music bed | r2, r6: none (no energy below 90 Hz). r1, r4, r5: a quiet bed under the voice. r3: bass impacts on reveals |
| SFX | Sparse. No whoosh on every zoom; a few impacts in r3 |
| Ducking | n/a where no bed; where a bed exists it sits far under the voice |

## 8. Branding and CTA

- Watermark only in r5: "@creator.adarshh", small, ~35 % opacity, mid-right. Others have none.
- CTAs: "comment.." white pill with blue border at top (r4); "COMMENT" gradient button + "Follow" button over the
  face-cam (r5); "Creator.adarsh · Follow" pill (r3).

---

## 9. What the pipeline will need from you per episode

1. Narration audio (as planned).
2. **Face-cam clip** for the outro or intercuts (you chose: you give it, we splice it in at the right place).
3. Nothing else for a normal episode. B-roll comes from free licensed stock (Pexels/Pixabay), never real people; a
   placeholder goes wherever a real person's photo would be.

## 10. Decisions (answered 2026-10-18)

- **Default look: A (metro_dark)**; B and C stay available as per-episode options.
- **Imagery: free open data**: Copernicus Sentinel-2 L2A composites + NASA Blue Marble + Copernicus DEM, with a
  tiny corner credit.
- **Fonts: free stand-ins**: Inter, Poppins, Playfair Display.

Background for each decision:

1. **Default look:** A (metro_dark)? Keep B and C as options?
2. **Satellite imagery source.** All three looks need real satellite/terrain imagery, and for a monetised page the
   licence has to allow commercial video. What I checked (2026-10-18):
   - **Mapbox Satellite / Esri World Imagery:** their attribution docs cover apps, print and static maps (credit
     "© Mapbox © OpenStreetMap © Maxar" / "Esri, Maxar, Earthstar Geographics"), but **neither says commercial video is
     allowed**. Usable only after you confirm with them in writing. Google Earth imagery has the same problem.
   - **EOX Sentinel-2 cloudless:** free only for non-commercial use (CC BY-NC-SA 4.0); commercial use needs a paid EOX
     licence. Out, unless you buy it.
   - **Recommended, free and commercial-safe:** build the imagery ourselves from open data.
     - Raw **Copernicus Sentinel-2 L2A** scenes (free incl. commercial use, credit "Contains modified Copernicus Sentinel
       data 2025"). The pipeline makes a cloud-free composite of each episode's area, cached offline. 10 m/px: sharp from
       country down to city scale (a ~10 km wide frame), soft at street level (r2's 42nd Street zoom would look blurry).
     - **NASA Blue Marble** (public domain) for world/continent zooms.
     - **Copernicus DEM GLO-30** (free incl. commercial, with credit) for look C's greyscale relief.
   - The references show **no map credit on screen**. Open data still needs a credit, so we'd add a tiny, low-opacity
     line in a corner like the r5 watermark. Every source and licence is logged in `data/geo/SOURCES.md`.
3. **Fonts:** do you have the editor's actual font files (SF Pro, etc.)? If not, Inter / Poppins / Playfair stand in.
