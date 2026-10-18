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

To be added with the imagery stage: Copernicus Sentinel-2 L2A, NASA Blue Marble, Copernicus DEM GLO-30
(see `style/STYLE.md` §10). Metro/railway lines, when an episode needs them: OpenStreetMap via Overpass
(ODbL, "© OpenStreetMap contributors"), cached per query.
