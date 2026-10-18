---
name: map-explainer-reel
engine: mapani-engine-kit
medium: 2d-vector
priority: 70
---
# Map-explainer reel (@creator.adarshh): satellite map + voiceover, keyword popups, face-cam outro

Machine values: `style/tokens.json`. Measured spec: `style/STYLE.md`.

## Recognition cues
- Frame: 9:16 top-down satellite (or greyscale relief) map, no tilt. The subject is highlighted by a darkened
  surround, a colour tint/flat fill, or a neon glow. Thick round-capped route lines draw on; small station dots with
  yellow or white label boxes.
- Edit: one continuous camera from frame 0 (zooms, no flyovers); hard cuts only; a new graphic every 1–2 s; outro is
  the creator on camera for 12–17 s (or a "comment.." pill).
- Sound: fast Hindi/Hinglish voiceover with no pauses over 0.35 s, −14 LUFS, heavy compression; music bed optional and
  quiet; almost no SFX.
- Text: no subtitles. Only typewriter-in keywords, yellow labels + neon-green numbers, titles at the top.

## Production defaults
- 1080×1920, 30 fps, 42–56 s. Look `metro_dark` unless the episode asks for another (`bright_political`, `hillshade_glow`).
- Each graphic starts 0.0–0.2 s after its word (typewriter 0.35 s); the camera starts on the place name (lead 0.1 s).
- Camera per look: A near-linear 3.5–6.5 s moves plus drift; B easeInOutSine 1.6–3 s; C easeOutCubic punch-ins 1.5–4 s.
- Real geography only (offline GeoJSON; India = SoI outline via DataMeet). Imagery from a licensed source, with a tiny credit.
- B-roll: licensed stock showing exactly what is said, with the stat typed on top. Never real people; use a placeholder.

## Known pitfalls
- Projected overlays drift if the basemap and the overlay use different cameras: draw both from one camera per frame.
- Natural Earth / world-atlas India borders are disputed-boundary versions: never use them for India; clip neighbours
  (Pakistan, China) to the SoI outline.
- Overseas territories blow up country bboxes (France + Guiana): keep the mainland plus nearby islands only.
- Labels must clear the highlighted shape, each other, and the IG safe area (top 210, bottom 400, right 140 px).
- Whisper drops words on pause-free Hinglish and stretches the rest over the gap, and large-v3 and large-v2 fail on
  different reels. Use temperature 0 (deterministic), no initial prompt, 12 s VAD chunks; detect broken stretches
  (< 3 words per 3 s of loud audio, or words > 1 s) and fill them from the other model (`pipeline/transcribe.py`).
- The face detector mistakes globes and news photos for face-cam; segment kinds come from the face-cam clip the user
  supplies, not from detection.
