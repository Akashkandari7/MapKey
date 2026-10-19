# Geo data: sources and licences

Every dataset the pipeline draws from. Raw downloads live in `data/geo/raw/` (not committed); `layers.gpkg` and
`names.json` are built from them by `python -m pipeline.geo.build_layers`.

| Dataset | Used for | Source | Licence | Credit |
|---|---|---|---|---|
| DataMeet `Country/india-composite.geojson` | India outline (Survey of India claim: all of J&K, Ladakh incl. Aksai Chin, Gilgit-Baltistan, Shaksgam) | https://github.com/datameet/maps (downloaded 2026-10-18) | CC0 (compiled from public-domain LSIB, Alhasan Systems, Natural Earth) | DataMeet India community |
| DataMeet `States/Admin2` | 36 Indian states/UTs (current: Ladakh separate, DNH & DD merged), clipped to the outline above | same repo | CC BY 4.0 (repo default) | "India boundaries by DataMeet India community (CC BY 4.0)" |
| Natural Earth 10m admin-0 countries, **India point of view** | All other countries (clipped so none overlaps SoI India) | https://www.naturalearthdata.com | Public domain | Natural Earth (optional) |
| Natural Earth 10m admin-1, rivers, lakes, geography regions, marine areas | World provinces, rivers, lakes, ranges/deserts/plateaus, seas/bays | same | Public domain | Natural Earth (optional) |
| GeoNames cities5000 + admin1 codes | Cities/towns ≥ 5000 people, with alternate names (incl. Devanagari) | https://download.geonames.org/export/dump/ (2026-10-18) | CC BY 4.0 | "GeoNames (geonames.org), CC BY 4.0" |

Not used, on purpose:
- DataMeet `india-soi.geojson`: CC BY-SA 2.5 / ODbL. Share-alike terms would complicate commercial reels; the CC0
  composite gives the same SoI outline.
- Raw Natural Earth / world-atlas default (non-IND) editions: they show disputed borders for India.

Checks run on the build (2026-10-18):
- Gilgit, Muzaffarabad, Aksai Chin and Shaksgam fall inside India; Gilgit and Aksai Chin are in Ladakh, Muzaffarabad in J&K
  (matches the 2019 SoI map).
- Overlap of PAK / CHN / NPL / BGD / MMR / BTN with India: 0.00 km².
- Unassigned land along India's land borders: 0.3 km² (sub-pixel); the remaining 4,433 km² difference is coastline
  generalisation (it renders as sea).
- 28 states + 8 union territories.

Planned / under-construction lines that OpenStreetMap doesn't have (e.g. the Gurugram Metro Millennium City-Cyber
City corridor, Oct 2026): the ordered station list comes from Gemini (cached in data/geo/cache/stations_*.json, with
its stated basis), each station is placed with OSM Nominatim inside the city, stations that break a 4.5 km
consecutive-station chain are dropped as name collisions (IMT Manesar's Sector 3/5/9), and the line is drawn only if
at least 65 % of the stations remain. Such lines are always listed under `needs_review` in geo.json. A line is never
drawn through guessed points; an official alignment file (KML/GeoJSON) given by the user would replace this.

Imagery (`pipeline/imagery.py`): Copernicus Sentinel-2 L2A median mosaics for city and region views (<= 160 m/px),
and NASA Blue Marble (shaded relief + bathymetry, public domain) for country and continent views (611-9784 m/px),
fetched as Web Mercator tiles from NASA GIBS (https://gibs.earthdata.nasa.gov, no login), cached in
`data/geo/cache/imagery/gibs/`. Neither carries borders: every border on screen comes from the layers above.
Land shapes for fills: a unit's land polygon from Natural Earth admin-1 / admin-0 is used instead of an OSM boundary
that includes sea (Hong Kong, Macau), then clipped to the Survey of India outline like everything else.
Still to add when needed: Copernicus DEM GLO-30 (see `style/STYLE.md` §10). Metro/railway lines, when an episode needs them: OpenStreetMap via Overpass
(ODbL, "© OpenStreetMap contributors"), cached per query.
