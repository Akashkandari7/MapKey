# Assets, fonts, code and data: sources and licences

Every third-party thing a rendered reel or this repo depends on. Geo datasets are detailed in `data/geo/SOURCES.md`.

| Item | Used for | Source | Licence | On-screen credit |
|---|---|---|---|---|
| Copernicus Sentinel-2 L2A true colour (via AWS open data, Element84 Earth Search catalogue) | Satellite basemap (median of clear scenes) | https://registry.opendata.aws/sentinel-2-l2a-cogs/ | Copernicus Sentinel data terms: free for any use incl. commercial, with attribution | "Contains modified Copernicus Sentinel data <years>" (added automatically) |
| NASA Blue Marble, shaded relief + bathymetry (via NASA GIBS tiles) | Basemap for country / continent views | https://gibs.earthdata.nasa.gov | NASA imagery: public domain (no copyright in the US) | "NASA Blue Marble (GIBS)" (added automatically) |
| Wikimedia Commons photos (picture cards) | A landmark photo of each place as it is named | https://commons.wikimedia.org | only public domain, CC0, CC BY, CC BY-SA; photos with people are rejected | author + licence printed on each card (added automatically) |
| OpenStreetMap (Nominatim, Overpass) | Metro/rail lines, roads, neighbourhoods, city boundaries | https://www.openstreetmap.org | ODbL 1.0 | "© OpenStreetMap contributors" (added automatically when used) |
| DataMeet India boundaries, Natural Earth, GeoNames | Countries, Indian states (Survey of India outline), places | see `data/geo/SOURCES.md` | CC0 / CC BY 4.0 / public domain | not required on screen (CC BY credit kept in SOURCES.md) |
| Inter (via @fontsource/inter) | Look A text (stand-in for SF Pro) | https://rsms.me/inter | SIL OFL 1.1 | none needed |
| Poppins (via @fontsource/poppins) | Look B text | https://fonts.google.com/specimen/Poppins | SIL OFL 1.1 | none needed |
| Playfair Display (via @fontsource/playfair-display) | Look C headlines | https://fonts.google.com/specimen/Playfair+Display | SIL OFL 1.1 | none needed |
| ReelMimic engine-kit `kit.js`, `render.mjs` | Frame runtime + parallel headless-Chrome renderer | https://github.com/edenfunf/reelmimic | MIT (`pipeline/render/LICENSE-*.txt`) | none needed |
| puppeteer-core | Drives headless Chrome | npm | Apache-2.0 | none needed |
| faster-whisper (large-v3 / large-v2 models) | Transcription | https://github.com/SYSTRAN/faster-whisper | MIT (models: MIT, OpenAI Whisper) | none needed |
| Google Gemini API | Entity extraction | https://ai.google.dev | Google API terms (paid/free tier) | none needed |

Not used: music, SFX or stock footage yet. Only files you put in `inputs/assets/`, or properly licensed sources
logged here, will be used.
