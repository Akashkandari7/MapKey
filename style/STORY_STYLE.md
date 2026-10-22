# Story format (@adarshh.gupta): measured style spec

Source: the 4 reels in `inputs/reference_reels/more better quality/` (q1 Mudra Yojana 72 s, q2 Delhi water 100 s,
q3 Nepal flood 66 s, q4 Haryana pollution 85 s; 720x1280, 24-30 fps). Measured with ffmpeg scene detection,
spectrograms, 2-s and 4-fps contact sheets and full-resolution frame crops. Machine-readable values:
`style/tokens.json` -> `story` (and `looks.story_map` for maps). The older map-only format is in `STYLE.md`.

## 1. What the format is

A reel is a chain of **beats of 2.5-6 s**, each a different picture that shows exactly what the voice says:

| Visual | Share of screen time | Seen in |
|---|---|---|
| AI photo / 3D illustration (always moving) | ~55 % | bakery, office, rain on India Gate, isometric building, cross-sections |
| Map (relief, coloured regions, labels, magnifier circles) | ~20 % | India heat map, Haryana districts, Nepal flood path |
| Real news clipping, marker sweep on the spoken line | ~12 % | ThePrint, Hindustan Times, Tribune, The Hindu |
| CTA paper + Instagram Follow card | last 6-10 s | all four |
| Creator on camera | short | q4 'WHAT IS THE REASON?' (cut-out over the map) |

## 2. Measurements

| Item | Value | How |
|---|---|---|
| Hard cuts | 9-21 per reel; the rest are 0.3-0.6 s dissolves, zoom-blur push-throughs, cloud reveals, defocus dissolves | `scdet` + 4-fps strips |
| Opening | falls through clouds onto a map (q2, q4) or a bold question over a photo (q3) | sheets |
| Headline text | Montserrat-like 900 caps, white line + yellow (or yellow->orange) line, soft dark shadow + colour glow, ~62 px at 720 | frame crops |
| Numbers | condensed bold yellow (~100 px at 720) between Poppins body lines, on a soft dark cloud | q2 '80%' |
| Map labels | black box, thin red top stripe, white sans 17 px at 720, pointer | q4 |
| Region look | relief/satellite, subject saturated (yellow/pink/green) with its texture visible, white edge | q4, q2 |
| Watermark | '@handle', Poppins 600 ~30 px at 720, white ~75 %, bottom centre (y 0.90-0.95), every shot | all |
| Finish | soft film grain + vignette on everything | crops |
| Sound | -14.2..-14.4 LUFS; music bed with deep bass all through; sub booms / whooshes on big transitions | `ebur128`, spectrograms |
| CTA | torn white paper strip mid-frame, question typing with the voice, key words red caps | q1, q2, q4 |
| End | dimmed IG profile, avatar in a blue ring, 'Follow' pressed by a hand cursor -> 'Following', '@handle' typing | q1, q3, q4 |

## 2b. House rules (from the user's review of our first q2 remake, 2026-10-22)

- Text never types: sentences come in word by word (fade up + soft blur), headlines pop in.
- The main place gets a **photo map**: its outline cut into the satellite map as a window onto the problem (q2 Delhi
  with a dry tap), ragged glowing red rim.
- Where something comes from is a **flow map**: rivers / canals / routes glowing cyan with light running along them
  into the place, an arrow at the tip, and an info box per source ('57% WATER / FROM HARYANA').
- AI photos carry the place's landmark (Delhi: India Gate) so the viewer knows where we are without text.
- A real person named in the voice is shown for real: the news card keeps the article's own photo.
- News cards show the paper's real masthead (its logo, captured from the page) on a printed-paper tint.
- Maps are colour-matched to the reference (green-olive satellite, Lab 104/117/136).

## 3. How the pipeline makes it

`python make_reel.py --audio <narration>` (story is the default mode):

1. **transcribe / entities / geocode**: as before (word timings, places, numbers, real geometry, India SoI rule).
2. **storyboard** (`pipeline/storyboard.py`, Gemini): beats, visual per beat, image prompts, map plan, news query,
   on-screen text, transition and effect. Hand-editable `work/<ep>/storyboard.json`.
3. **visuals** (`pipeline/visuals.py`): Gemini image model at 9:16 2K (recurring people keep their face via a
   reference picture); news clippings from `inputs/<ep>/news.txt` links, or `--find-news` search (every hit is checked
   by Gemini against what the voice says; a contradicting headline is rejected); end card from `inputs/brand/`.
4. **plan** (`pipeline/story_plan.py`): shots, camera dives, coloured regions, tags, magnifiers, text timing,
   number count-ups, transitions.
5. **review**: India rule on every map shape, missing pictures, text wider than the frame (auto-scaled), long holds,
   list of news links to check.
6. **render** (`pipeline/render/story.js` on the mapani engine) and **mix** (`pipeline/sound.py`: Lyria music bed
   ducked under the voice + synthesised whoosh / boom / riser on the cuts; whole mix to -14 LUFS).

Per creator: `inputs/brand/avatar.jpg` (end card photo), `inputs/brand/brand.json` (`{"handle": "@name"}`),
optional `inputs/brand/profile.jpg` (dimmed backdrop of the end card).
