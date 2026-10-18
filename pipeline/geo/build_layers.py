"""Build the offline geo database the geocoder reads: data/geo/layers.gpkg + data/geo/names.json.

    python -m pipeline.geo.build_layers            (re-run only when data/geo/raw changes)

Layers (all EPSG:4326):
    countries  Natural Earth 10m admin-0, India's point-of-view edition. India's polygon is REPLACED by the DataMeet
               india-composite outline (Survey of India claim: all of J&K, Ladakh incl. Aksai Chin, Gilgit-Baltistan,
               Shaksgam). Every other country is clipped by that outline, so no neighbour ever draws over India.
    admin1     Natural Earth 10m admin-1 for the world, except India, which comes from DataMeet's 36 states/UTs
               (current: Ladakh separate, DNH & DD merged), clipped to the India outline. Hindi names from NE.
    places     GeoNames cities5000 (with alternate names incl. Devanagari) + Natural Earth populated places.
    rivers, lakes, regions (mountain ranges, deserts, plateaus...), marine (seas, bays, gulfs): Natural Earth 10m.

names.json: normalised name -> [{layer, fid, name, rank}] for fast lookup; rank = importance (higher wins ties).
Sources and licences: data/geo/SOURCES.md.
"""
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

RAW = Path("data/geo/raw")
OUT_GPKG = Path("data/geo/layers.gpkg")
OUT_NAMES = Path("data/geo/names.json")

# DataMeet spellings -> Natural Earth spellings (for pulling Hindi names / codes across)
DM_TO_NE = {"Jammu & Kashmir": "Jammu and Kashmir", "Andaman & Nicobar": "Andaman and Nicobar",
            "Dadra and Nagar Haveli and Daman and Diu": "Dadra and Nagar Haveli", "Delhi": "Delhi",
            "Puducherry": "Puducherry", "Odisha": "Odisha", "Uttarakhand": "Uttarakhand"}
# Hindi names NE lacks or has outdated
HI_EXTRA = {"Ladakh": "लद्दाख", "Telangana": "तेलंगाना", "Dadra and Nagar Haveli and Daman and Diu": "दादरा और नगर हवेली और दमन और दीव",
            "Jammu & Kashmir": "जम्मू और कश्मीर", "Andaman & Nicobar": "अंडमान और निकोबार"}
# Natural Earth's Indian state types are partly outdated (it calls Himachal Pradesh a UT); this list is authoritative
INDIA_UTS = {"Andaman & Nicobar", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
             "Jammu & Kashmir", "Ladakh", "Lakshadweep", "Puducherry"}
# common alternate names people say (English/Hinglish)
ALIASES = {"Gurgaon": ["Gurugram"], "Bengaluru": ["Bangalore"], "Mumbai": ["Bombay"], "Kolkata": ["Calcutta"],
           "Chennai": ["Madras"], "Prayagraj": ["Allahabad"], "Varanasi": ["Banaras", "Benares", "Kashi"],
           "Puducherry": ["Pondicherry"], "Odisha": ["Orissa"], "Uttarakhand": ["Uttaranchal"],
           "Myanmar": ["Burma"], "Sri Lanka": ["Ceylon"], "United States of America": ["USA", "US", "America"],
           "United Kingdom": ["UK", "Britain", "England"], "Jammu & Kashmir": ["J&K", "Kashmir"]}


def norm(s):
    """Lower-case, strip accents and punctuation; keep Devanagari letters and matras intact."""
    if not isinstance(s, str):
        return ""
    s = s.replace("&", " and ")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not (unicodedata.combining(c) and ord(c) < 0x900))
    s = re.sub(r"[^\wऀ-ॿ]+", " ", s.casefold())
    return re.sub(r"\s+", " ", s).strip()


def read_zip(name):
    return gpd.read_file(f"zip://{RAW / name}")


def build():
    india = gpd.read_file(RAW / "india-composite.geojson").geometry.make_valid().union_all()

    # countries
    c = read_zip("ne_10m_admin_0_countries_ind.zip")
    c = c[["ADM0_A3", "ISO_A2", "NAME", "NAME_LONG", "NAME_EN", "NAME_HI", "ADMIN", "SUBREGION", "MIN_ZOOM", "geometry"]].copy()
    c["geometry"] = c.geometry.make_valid()
    is_ind = c.ADM0_A3 == "IND"
    c.loc[~is_ind, "geometry"] = c.loc[~is_ind, "geometry"].difference(india)
    c.loc[is_ind, "geometry"] = gpd.GeoSeries([india] * is_ind.sum(), index=c.index[is_ind], crs=c.crs)
    # Land that Natural Earth's India polygon covers but the SoI outline doesn't (thin slivers along the China/Nepal/
    # Bangladesh borders, ~35k km2 in total) would otherwise belong to no country and show as gaps when a neighbour is
    # filled. Give each sliver to the neighbour it shares the longest edge with.
    ne_india = read_zip("ne_10m_admin_0_countries_ind.zip").query("ADM0_A3 == 'IND'").geometry.make_valid().union_all()
    leftover = ne_india.difference(india)
    parts = [p for p in getattr(leftover, "geoms", [leftover]) if p.geom_type == "Polygon" and p.area > 1e-7]
    near = c.index[~is_ind & c.geometry.intersects(india.buffer(1.0))]  # only India's neighbours can touch a sliver
    gained = {k: [] for k in near}
    for part in parts:
        ring = part.buffer(0.01)
        shared = c.loc[near, "geometry"].intersection(ring).area
        if len(shared) and shared.max() > 0:
            gained[shared.idxmax()].append(part)
    for k, ps in gained.items():
        if ps:
            c.loc[k, "geometry"] = c.loc[k, "geometry"].union(gpd.GeoSeries(ps).union_all()).difference(india)
    c = c[~c.geometry.is_empty]
    c["source"] = c.ADM0_A3.map(lambda a: "DataMeet india-composite (SoI)" if a == "IND" else "Natural Earth 10m (IND view), clipped to SoI India")
    c["rank"] = 100 - c.MIN_ZOOM.fillna(5) * 5

    # admin1
    a = read_zip("ne_10m_admin_1_states_provinces.zip")
    a = a[["adm0_a3", "iso_a2", "name", "name_en", "name_hi", "type_en", "region", "min_zoom", "geometry"]].copy()
    ne_ind = a[a.adm0_a3 == "IND"].set_index("name")
    a = a[a.adm0_a3 != "IND"]
    a["geometry"] = a.geometry.make_valid().difference(india)  # foreign provinces never overlap India
    dm = gpd.read_file(RAW / "dm_states_Admin2.shp").to_crs(4326)
    dm["geometry"] = dm.geometry.make_valid().intersection(india)
    rows = []
    for _, r in dm.iterrows():
        ne_name = DM_TO_NE.get(r.ST_NM, r.ST_NM)
        hi = HI_EXTRA.get(r.ST_NM) or (ne_ind.loc[ne_name, "name_hi"] if ne_name in ne_ind.index else None)
        typ = "Union Territory" if r.ST_NM in INDIA_UTS else "State"
        rows.append({"adm0_a3": "IND", "iso_a2": "IN", "name": r.ST_NM, "name_en": r.ST_NM, "name_hi": hi,
                     "type_en": typ, "region": None, "min_zoom": 3.0, "geometry": r.geometry})
    a = pd.concat([a, gpd.GeoDataFrame(rows, crs=4326)], ignore_index=True)
    a = gpd.GeoDataFrame(a, crs=4326)
    a = a[~a.geometry.is_empty]
    a["source"] = a.adm0_a3.map(lambda x: "DataMeet States (SoI-aligned), clipped" if x == "IND" else "Natural Earth 10m admin-1")
    a["rank"] = 80 - a.min_zoom.fillna(6) * 4

    # places: GeoNames cities5000 (+ alternate names) and NE populated places
    cols = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass", "fcode", "cc", "cc2",
            "admin1", "admin2", "admin3", "admin4", "population", "elevation", "dem", "tz", "moddate"]
    with zipfile.ZipFile(RAW / "geonames_cities5000.zip") as z:
        g = pd.read_csv(z.open("cities5000.txt"), sep="\t", header=None, names=cols, dtype={"admin1": str},
                        keep_default_na=False, quoting=3)
    adm1 = pd.read_csv(RAW / "geonames_admin1CodesASCII.txt", sep="\t", header=None, names=["code", "n", "na", "id"],
                       keep_default_na=False)
    adm1n = dict(zip(adm1.code, adm1.na))
    g["adm1name"] = [adm1n.get(f"{cc}.{a1}", "") for cc, a1 in zip(g.cc, g.admin1)]
    places = gpd.GeoDataFrame({
        "name": g.name, "asciiname": g.asciiname, "alternatenames": g.alternatenames, "cc": g.cc,
        "adm1name": g.adm1name, "population": g.population, "fcode": g.fcode,
        "source": "GeoNames cities5000"}, geometry=[Point(xy) for xy in zip(g.lon, g.lat)], crs=4326)
    places["rank"] = places.population.clip(lower=1).map(lambda p: 20 + 6 * (len(str(int(p))) - 1))
    # cities like Gilgit/Muzaffarabad sit inside SoI India: GeoNames codes them PK; keep the name, flag the claim
    places["in_soi_india"] = places.within(india)

    layers = {"countries": c, "admin1": a, "places": places}
    for key, fname, keep in [("rivers", "ne_10m_rivers_lake_centerlines.zip", ["name", "name_en", "name_hi", "featurecla", "min_zoom"]),
                             ("lakes", "ne_10m_lakes.zip", ["name", "name_en", "name_hi", "featurecla", "min_zoom"]),
                             ("regions", "ne_10m_geography_regions_polys.zip", ["NAME", "NAME_EN", "NAME_HI", "FEATURECLA", "REGION", "SCALERANK"]),
                             ("marine", "ne_10m_geography_marine_polys.zip", ["name", "name_en", "name_hi", "featurecla", "scalerank"])]:
        df = read_zip(fname)
        df = df[[k for k in keep if k in df.columns] + ["geometry"]].copy()
        df.columns = [k.lower() if k != "geometry" else k for k in df.columns]
        df = df[df["name"].notna() & (df["name"] != "")]
        df["source"] = "Natural Earth 10m"
        z = df["min_zoom"] if "min_zoom" in df else df.get("scalerank", 6)
        df["rank"] = 60 - pd.to_numeric(z, errors="coerce").fillna(6) * 4
        layers[key] = df

    OUT_GPKG.unlink(missing_ok=True)
    for key, df in layers.items():
        df = df.reset_index(drop=True)
        df["fid_"] = df.index
        df.to_file(OUT_GPKG, layer=key, driver="GPKG")
        layers[key] = df
        print(f"{key:10s} {len(df):6d} features")

    # name index
    idx = {}

    def add(n, layer, fid, display, rank):
        k = norm(n)
        if len(k) < 2:
            return
        e = {"layer": layer, "fid": int(fid), "name": display, "rank": round(float(rank), 1)}
        # both "tamil nadu" and "tamilnadu": spoken/transcribed names split words inconsistently
        for key in {k, k.replace(" ", "")}:
            idx.setdefault(key, []).append(e)

    for layer, df in layers.items():
        name_cols = [x for x in ("name", "name_en", "name_long", "name_hi", "admin", "asciiname") if x in [k.lower() for k in df.columns]]
        lower = {k.lower(): k for k in df.columns}
        for _, r in df.iterrows():
            disp = r[lower["name"]]
            names = {r[lower[x]] for x in name_cols if isinstance(r[lower[x]], str)}
            if layer == "places" and r["alternatenames"]:
                for alt in r["alternatenames"].split(","):
                    if re.search(r"[A-Za-zऀ-ॿ]", alt) and not re.search(r"[^\x00-ɏऀ-ॿ\s'.-]", alt):
                        names.add(alt)
            for al in ALIASES.get(disp, []):
                names.add(al)
            for n in names:
                add(n, layer, r["fid_"], disp, r["rank"])
    for k in idx:
        idx[k].sort(key=lambda e: -e["rank"])
    OUT_NAMES.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    print(f"names.json: {len(idx)} names")


if __name__ == "__main__":
    build()
