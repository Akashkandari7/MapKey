"""Stage 3: entities.json -> geo.json (real geometry for every place and route).

    python -m pipeline.geocode work/<episode>/entities.json --out work/<episode>/geo.json [--offline]

Order of sources for each place (first good match wins):
  1. Offline database data/geo/layers.gpkg (pipeline/geo/build_layers.py): countries with the Survey of India outline,
     Indian states (DataMeet), world provinces, 70k cities/towns, rivers, lakes, seas, mountain ranges...
     Candidates come from names.json (English + Hindi + alternate names) and are scored on: layer fits the entity kind,
     country/parent match, distance to the LLM's approximate lat/lon, and importance.
  2. OpenStreetMap Nominatim (online, cached in data/geo/cache/), for things the offline data doesn't have (stations,
     landmarks, neighbourhoods) and for city boundary polygons when the city is to be highlighted.
Routes (metro lines, highways, railways): OpenStreetMap Overpass by name/ref (cached); if none is found, the route is
the line through its stops, flagged "approx" so the review loop and you can see it.

India rule: any geometry inside India is drawn from the SoI-based layers. Nominatim/OSM results are clipped to the SoI
India outline only when they are Indian sub-areas, and countries never come from OSM.

Output per entity: resolved source + confidence, GeoJSON geometry (simplified to the zoom it will be seen at), bbox,
label point. Plus "context": the polygons the renderer needs around the story (main country + neighbours, its states).
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
from shapely.ops import linemerge, unary_union

from pipeline.common import ROOT, read_json, sha, write_json
from pipeline.geo.build_layers import norm

GPKG = ROOT / "data/geo/layers.gpkg"
NAMES = ROOT / "data/geo/names.json"
CACHE = ROOT / "data/geo/cache"
UA = "MapaniReelPipeline/0.1 (offline-first map-explainer renderer; low volume, cached)"

KIND_LAYERS = {
    "country": ["countries"], "continent": ["regions"], "state": ["admin1"], "district": ["admin1", "places"],
    "city": ["places"], "neighbourhood": ["places"], "river": ["rivers"], "lake": ["lakes"],
    "sea": ["marine"], "ocean": ["marine"], "mountain": ["regions"], "mountain_range": ["regions"],
    "region": ["regions", "admin1"], "desert": ["regions"], "island": ["regions", "places"],
    "border": ["countries"],
}
ONLINE_FIRST = {"landmark", "road", "highway", "metro_line", "metro_station", "railway_line", "railway_station",
                "airport", "port", "neighbourhood"}
# how far (km) the LLM's lat/lon hint may be from a correct match, by kind
HINT_KM = {"country": 1500, "continent": 3000, "state": 400, "region": 800, "mountain_range": 800, "desert": 800,
           "sea": 1200, "ocean": 3000, "river": 600, "lake": 150, "city": 60, "district": 80, "island": 200,
           # small things: names repeat across a metro area ("Udyog Vihar" exists in Gurugram AND Greater Noida)
           "neighbourhood": 12, "metro_station": 6, "railway_station": 20, "landmark": 25, "airport": 40, "port": 40}


def haversine(lon1, lat1, lon2, lat2):
    p = math.pi / 180
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(a))


class GeoDB:
    def __init__(self):
        self.names = json.loads(NAMES.read_text(encoding="utf-8"))
        self.layers = {}
        self.india = None

    def layer(self, name):
        if name not in self.layers:
            self.layers[name] = gpd.read_file(GPKG, layer=name).set_index("fid_")
        return self.layers[name]

    def india_outline(self):
        if self.india is None:
            c = self.layer("countries")
            self.india = c[c.ADM0_A3 == "IND"].geometry.iloc[0]
        return self.india

    def candidates(self, *names):
        seen, out = set(), []
        for n in names:
            k = norm(n or "")
            for key in (k, k.replace(" ", "")):
                for e in self.names.get(key, []):
                    t = (e["layer"], e["fid"])
                    if t not in seen:
                        seen.add(t)
                        out.append(e)
        return out


def country_of(row, layer):
    for col in ("ADM0_A3", "adm0_a3"):
        if col in row and isinstance(row[col], str):
            return row[col]
    if layer == "places":
        return row.get("cc")
    return None


ISO2_TO_A3 = None


def iso2_to_a3(db, iso2):
    global ISO2_TO_A3
    if ISO2_TO_A3 is None:
        c = db.layer("countries")
        ISO2_TO_A3 = dict(zip(c.ISO_A2, c.ADM0_A3))
        ISO2_TO_A3["IN"] = "IND"
    return ISO2_TO_A3.get((iso2 or "").upper())


def score(db, ent, cand):
    row = db.layer(cand["layer"]).loc[cand["fid"]]
    geom = row.geometry
    kind = ent.get("kind")
    s = cand["rank"] / 10.0
    if kind and cand["layer"] in KIND_LAYERS.get(kind, []):
        s += 6 - KIND_LAYERS[kind].index(cand["layer"])
    elif kind:
        s -= 4
    want = ent.get("country")
    if want:
        got = country_of(row, cand["layer"])
        a3 = iso2_to_a3(db, want)
        if got in (want.upper(), a3):
            s += 3
        elif cand["layer"] == "places" and row.get("in_soi_india") and want.upper() == "IN":
            s += 3  # GeoNames codes Gilgit etc. as PK; SoI says India
        elif got:
            s -= 3
    parent = norm(ent.get("parent") or "")
    if parent and cand["layer"] == "places" and parent in norm(str(row.get("adm1name", ""))):
        s += 2
    dist = None
    if ent.get("lat") is not None and ent.get("lon") is not None:
        rp = geom.representative_point() if geom.geom_type != "Point" else geom
        dist = haversine(ent["lon"], ent["lat"], rp.x, rp.y)
        if geom.geom_type != "Point" and geom.buffer(0.05).contains(Point(ent["lon"], ent["lat"])):
            dist = 0.0
        lim = HINT_KM.get(kind, 300)
        s += 4 if dist <= lim else (-2 if dist <= 3 * lim else -8)
    return s, dist, row


def _nominatim_get(endpoint, params):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"nominatim_{sha(endpoint, params)}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    url = f"https://nominatim.openstreetmap.org/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    time.sleep(1.1)  # usage policy: max 1 request/s
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def nominatim(q, polygon=True, countrycodes=None):
    params = {"q": q, "format": "geojson", "limit": 5, "polygon_geojson": 1 if polygon else 0, "polygon_threshold": 0.0005,
              "addressdetails": 1, "accept-language": "en"}
    if countrycodes:
        params["countrycodes"] = countrycodes.lower()
    return _nominatim_get("search", params)


def nominatim_reverse(lon, lat, zoom):
    params = {"lat": round(lat, 5), "lon": round(lon, 5), "zoom": zoom, "format": "geojson", "polygon_geojson": 1,
              "polygon_threshold": 0.0005, "accept-language": "en"}
    return _nominatim_get("reverse", params).get("features", [])


def city_polygon(ent, point, first=None):
    """The boundary to highlight for a city: among OSM admin areas around the city's point whose name is exactly the
    city's name (old/new spellings equal), the smallest one of city size. Gurugram: the 333 km2 'Gurgaon' city
    district, not the 1253 km2 Gurugram district (what the reference reel highlights); 'Delhi' never becomes the
    'New Delhi' district because the names must be equal, not just overlap."""
    want = set(tokens(bare_name(ent)))
    cands = [first] if first else []
    for z in (10, 8, 6):
        try:
            cands += nominatim_reverse(point[0], point[1], z)
        except Exception as e:  # noqa: BLE001
            print(f"[geocode] reverse failed at zoom {z}: {e}", file=sys.stderr)
    best = None
    for f in cands:
        g = shape(f["geometry"])
        if g.geom_type not in ("Polygon", "MultiPolygon") or not g.buffer(0.001).contains(Point(point)):
            continue
        if set(tokens(f["properties"].get("name", ""))) != want:
            continue
        a = area_km2(g)
        if 20 <= a <= 1e4 and (best is None or a < best[0]):
            best = (a, f, g)
    return best


OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]


def overpass(query):
    """Run an Overpass query (cached). Busy servers answer 429/502/504: retry with backoff, then try the next mirror."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"overpass_{sha(query)}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    last = None
    for url in OVERPASS[:2]:
        for wait in (1, 5):
            time.sleep(wait)
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.loads(r.read().decode("utf-8"))
                f.write_text(json.dumps(data), encoding="utf-8")
                return data
            except Exception as e:  # noqa: BLE001  (network errors of every kind -> retry)
                last = e
    raise RuntimeError(f"all Overpass mirrors failed: {last}")


def simplify_for_view(geom):
    """Simplify to ~1/1500 of the feature's size (invisible at the zoom the feature fills the frame)."""
    if geom.geom_type == "Point":
        return geom
    minx, miny, maxx, maxy = geom.bounds
    tol = max(maxx - minx, maxy - miny) / 1500
    return geom.simplify(tol, preserve_topology=True) if tol > 0 else geom


def pack(geom, source, method, confidence, name, extra=None):
    g = simplify_for_view(geom)
    lp = geom if geom.geom_type == "Point" else geom.representative_point()
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        # polylabel-ish: the representative point is always inside; good enough for labels
        lp = geom.representative_point()
    d = {"name": name, "source": source, "method": method, "confidence": round(confidence, 2),
         "bbox": [round(v, 5) for v in geom.bounds], "label_point": [round(lp.x, 5), round(lp.y, 5)],
         "geometry": mapping(g)}
    if extra:
        d.update(extra)
    return d


# Plausible area (km2) of a correct online match, by kind. Rejects e.g. the village "Zealandia" (5 km2) returned
# for the sunken continent. Kinds not listed accept anything (points, lines, small places).
AREA_KM2 = {"continent": (1e6, None), "ocean": (1e6, None), "sea": (5e3, None), "country": (0.4, None),
            "state": (20, 5e6), "region": (500, None), "mountain_range": (500, None), "desert": (500, None),
            "lake": (0.2, None), "island": (0.1, None), "city": (2, 1e4), "district": (20, 5e4)}
NEEDS_AREA = {"continent", "ocean", "sea", "state", "region", "mountain_range", "desert", "lake", "district"}


def area_km2(g):
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        return 0.0
    return float(gpd.GeoSeries([g], crs=4326).to_crs(6933).area.iloc[0] / 1e6)


def plausible(kind, g):
    lo, hi = AREA_KM2.get(kind, (None, None))
    if lo is None:
        return True
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        return kind not in NEEDS_AREA
    a = area_km2(g)
    return a >= lo and (hi is None or a <= hi)


SMALL_KINDS = {"neighbourhood", "metro_station", "railway_station", "landmark", "airport", "port"}
AREA_KINDS = {"neighbourhood", "city", "district", "state", "country", "region", "island", "continent", "desert",
              "mountain_range"}

KIND_WORDS = {"metro_station": ["metro station", "station"], "railway_station": ["railway station", "junction", "station"],
              "airport": ["airport", "international airport"], "landmark": [], "neighbourhood": []}


# old/new and English/local spellings that name the same place; matching treats each pair as one word
CANON = {"gurgaon": "gurugram", "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
         "orissa": "odisha", "pondicherry": "puducherry", "allahabad": "prayagraj", "banaras": "varanasi",
         "benares": "varanasi", "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
         "cochin": "kochi", "simla": "shimla", "mysore": "mysuru", "centre": "center"}
SWAPS = [("Gurugram", "Gurgaon"), ("Bengaluru", "Bangalore"), ("Mumbai", "Bombay"), ("Kolkata", "Calcutta"),
         ("Chennai", "Madras"), ("Odisha", "Orissa"), ("Puducherry", "Pondicherry"), ("Prayagraj", "Allahabad")]


def tokens(s):
    return [CANON.get(t, t) for t in norm(s).split()]


def bare_name(ent):
    name = (ent.get("name_en") or "").strip()
    for w in sorted(KIND_WORDS.get(ent.get("kind"), []), key=len, reverse=True):
        if name.lower().endswith(" " + w):
            name = name[: -len(w) - 1].strip()
    return name


def name_match(want, got):
    """Every word of the wanted name appears in the result's name (fuzzy per word, old/new spellings equal).
    'Guru Dronacharya' ~ 'Guru Dronacharya'; 'Gurugram' ~ 'Gurgaon'; 'Old Gurugram' !~ 'Old Railway Road'."""
    import difflib
    w, g = tokens(want), tokens(got)
    if not w:
        return True
    hit = sum(1 for t in w if any(t == u or difflib.SequenceMatcher(None, t, u).ratio() >= 0.8 for u in g))
    # all wanted words present, and not buried in a longer name ("Old Gurgaon" is not "Old Delhi Gurgaon Road Flyover");
    # +2 words allowed for suffixes like "Metro Station", "Island"
    return (hit == len(w) and len(g) <= len(w) + 2) or difflib.SequenceMatcher(None, " ".join(w), " ".join(g)).ratio() >= 0.85


def nominatim_queries(ent):
    """'Guru Dronacharya metro station' + parent 'Gurugram' -> with parent, without, bare name + kind word, bare name,
    then the same with old/new spellings swapped (Gurugram <-> Gurgaon)."""
    name, parent = (ent.get("name_en") or "").strip(), (ent.get("parent") or "").strip()
    bare = bare_name(ent)
    qs = []
    # the reel's main city is better context than a state-level parent ("Udyog Vihar, Gurugram" vs "..., Haryana")
    for c in ent.get("_context", []):
        if c and norm(c) not in norm(name):
            qs += [f"{name}, {c}", f"{bare}, {c}"]
    qs += [f"{name}, {parent}" if parent else name, name]
    qs += [f"{bare} {w}" for w in KIND_WORDS.get(ent.get("kind"), [])]
    qs += [f"{bare}, {parent}" if parent else bare, bare]
    for a, b in SWAPS:
        for x, y in ((a, b), (b, a)):
            if x in bare:
                qs.append(bare.replace(x, y))
    out = []
    for q in qs:
        if q and q not in out:
            out.append(q)
    return out


def resolve_place(db, ent, offline=False):
    kind = ent.get("kind")
    names = [ent.get("name_en"), ent.get("quote")]
    best = None
    if kind not in ONLINE_FIRST:
        for cand in db.candidates(*names):
            s, dist, row = score(db, ent, cand)
            if best is None or s > best[0]:
                best = (s, dist, row, cand)
    want_poly = kind in ("city", "district") and ent.get("visual") in ("highlight", "zoom")
    if best and best[0] >= 8 and not (want_poly and best[2].geometry.geom_type == "Point" and not offline):
        s, dist, row, cand = best
        conf = min(1.0, 0.5 + s / 30)
        return pack(row.geometry, row.get("source", cand["layer"]), f"offline:{cand['layer']}", conf, cand["name"],
                    {"hint_km": round(dist, 1) if dist is not None else None})
    if offline:
        if best:
            s, dist, row, cand = best
            return pack(row.geometry, row.get("source", cand["layer"]), f"offline:{cand['layer']}:weak", 0.3,
                        cand["name"], {"hint_km": round(dist, 1) if dist is not None else None})
        return None
    # online: Nominatim, trying name variants; a result must match the name, the kind's size and the LLM's location
    want = bare_name(ent)
    for q in nominatim_queries(ent):
        try:
            feats = nominatim(q, polygon=True, countrycodes=ent.get("country")).get("features", [])
        except Exception as e:  # noqa: BLE001
            print(f"[geocode] Nominatim failed for {q!r}: {e}", file=sys.stderr)
            continue
        parent_toks = set(tokens(ent.get("parent") or ""))

        def rank(f):
            # results inside the named parent ("..., Gurugram, Haryana") first, then nearest to the LLM's location
            in_parent = bool(parent_toks) and parent_toks <= set(tokens(f["properties"].get("display_name", "")))
            d = haversine(ent["lon"], ent["lat"], *shape(f["geometry"]).representative_point().coords[0]) \
                if ent.get("lat") is not None else 0
            return (0 if in_parent else 1, d)
        feats.sort(key=rank)
        for f in feats:
            got = f["properties"].get("name") or f["properties"].get("display_name", "").split(",")[0]
            if not name_match(want, got):
                continue  # e.g. "Old Gurugram" must not resolve to "Old Railway Road"
            proxy = False
            if kind in AREA_KINDS and f["properties"].get("category") in ("highway", "railway", "building", "amenity", "shop", "office"):
                # an area/town must not resolve to a road, flyover or building of the same name - but when the area is
                # only pinned/labelled, a same-named bus stop or station inside it marks the spot well enough
                if not (ent.get("visual") in ("pin", "label") and f["geometry"]["type"] == "Point"
                        and f["properties"].get("type") in ("bus_stop", "station", "stop", "halt", "platform")):
                    continue
                proxy = True
            g = shape(f["geometry"])
            if not plausible(kind, g):
                continue
            if ent.get("lat") is not None:
                rp = g.representative_point()
                slack = 1 if kind in SMALL_KINDS else 2  # LLM coords for a neighbourhood/station are good to ~10 km
                if haversine(ent["lon"], ent["lat"], rp.x, rp.y) > HINT_KM.get(kind, 300) * slack and not g.buffer(0.01).contains(Point(ent["lon"], ent["lat"])):
                    continue
            if kind in ("city", "district") and ent.get("visual") in ("highlight", "zoom"):
                pt = (ent["lon"], ent["lat"]) if ent.get("lat") is not None else g.representative_point().coords[0]
                if not g.buffer(0.01).contains(Point(pt)):
                    pt = g.representative_point().coords[0]
                tight = city_polygon(ent, pt, first=f)
                if tight:
                    f, g = tight[1], tight[2]
                    got = f["properties"].get("name", got)
            if g.geom_type in ("Polygon", "MultiPolygon") and (ent.get("country") or "").upper() == "IN":
                g = g.intersection(db.india_outline())  # Indian sub-areas never cross the SoI outline
            return pack(g, "OpenStreetMap Nominatim (ODbL)", "nominatim:proxy" if proxy else "nominatim",
                        0.55 if proxy else 0.75, got,
                        {"osm": f"{f['properties'].get('osm_type')}/{f['properties'].get('osm_id')}", "query": q,
                         "area_km2": round(area_km2(g), 1)})
    if best:  # weak offline match beats nothing (e.g. city point when no polygon was found online)
        s, dist, row, cand = best
        return pack(row.geometry, row.get("source", cand["layer"]), f"offline:{cand['layer']}:fallback", 0.45, cand["name"],
                    {"hint_km": round(dist, 1) if dist is not None else None})
    if ent.get("lat") is not None:
        return pack(Point(ent["lon"], ent["lat"]), "LLM approximate coordinates", "llm_hint", 0.2, ent.get("name_en"))
    return None


def overpass_route(ent, around, country=None):
    """Find a named route (metro/rail line, highway ref) near `around` (lon, lat) in OSM."""
    name = ent.get("name_en") or ""
    mode = ent.get("mode")
    lon, lat = around
    bbox = f"{lat - 1.5},{lon - 1.5},{lat + 1.5},{lon + 1.5}"
    is_ref = mode == "road" and re.match(r"^(NH|National Highway|SH|State Highway|AH|I-|US)\s*-?\s*\d+", name, re.I)
    if is_ref:
        ref = re.sub(r"(?i)national highway", "NH", name).replace(" ", "").upper()
        ref_spaced = re.sub(r"^([A-Z]+)", r"\1 ", ref)
        # the same ref exists in several countries (NH44: India, Nepal, Myanmar) -> filter by network prefix
        net = f'["network"~"^{country.upper()}:"]' if country else ""
        selectors = [(f'relation["route"="road"]["ref"~"^({ref}|{ref_spaced})$"]{net}', None)]
    elif mode == "road":
        # named roads (Dwarka Expressway): a route relation if one exists, else the highway ways with that name
        selectors = [(f'relation["route"="road"]["name"~"{k}",i]({bbox})', k) for k in route_keys(name)[:2]]
        selectors += [(f'way["highway"]["name"~"{k}",i]({bbox})', k) for k in route_keys(name)[:2]]
    else:
        route_types = "subway|light_rail|train|monorail|tram" if mode in ("metro", "rail") else "subway|light_rail|train|bus|ferry"
        selectors = [(f'relation["route"~"^({route_types})$"]["name"~"{k}",i]({bbox})', k) for k in route_keys(name)]
    for sel, key in selectors:
        if sel.startswith("way"):
            q = f"[out:json][timeout:120];{sel};out tags geom;"
        else:
            # Long roads are super-relations (NH44 = 13 state sections): take child relations too, then only the
            # carriageway members (role "", forward, backward) - not spur ways tagged with descriptive roles.
            q = (f'[out:json][timeout:180];{sel}->.top;.top out tags;rel(r.top)->.kids;(.top;.kids;)->.all;'
                 f'(way(r.all:"");way(r.all:"forward");way(r.all:"backward");way(r.all:"main"););out geom;')
        data = overpass(q)
        els = data.get("elements", [])
        if key is not None:
            # the matched relation (or way) must carry the route's distinctive words, looking at name + network +
            # operator: "Gurugram Metro Extension" must not be satisfied by Rapid Metro, nor by Delhi Metro's
            # "Magenta Line (... Krishna Park Extension)" just because the loose search key "Extension" hit it
            kind = "way" if sel.startswith("way") else "relation"
            label = lambda e: " ".join(e.get("tags", {}).get(t, "") for t in ("name", "network", "operator"))
            if not any(route_name_match(name, label(e)) for e in els if e.get("type") == kind):
                continue
            if kind == "way":
                els = [e for e in els if route_name_match(name, label(e))]
        lines = [LineString([(p["lon"], p["lat"]) for p in el["geometry"]])
                 for el in els if el.get("type") == "way" and len(el.get("geometry", [])) >= 2]
        if lines:
            return linemerge(MultiLineString(lines))
    return None


def route_name_match(want, got):
    """The distinctive words of a route name must all be in the OSM name: 'Delhi Metro Yellow Line' needs 'yellow',
    'Rapid Metro Gurugram' needs 'rapid', 'Gurugram Metro Extension' needs 'extension'."""
    import difflib
    core = [t for t in tokens(want) if t not in ROUTE_FILLER]
    g = tokens(got)
    if not core:
        return name_match(want, got)
    return all(any(t == u or difflib.SequenceMatcher(None, t, u).ratio() >= 0.8 for u in g) for t in core)


# words that don't identify a route (city names DO identify it, and are checked against name + network + operator)
ROUTE_FILLER = {"metro", "metrorail", "line", "the", "of", "railway", "rail", "subway", "corridor", "route", "road", "new"}


GENERIC = {"metro", "line", "delhi", "mumbai", "bengaluru", "bangalore", "kolkata", "chennai", "hyderabad", "the", "of",
           "railway", "rail", "subway", "corridor", "gurgaon", "gurugram", "nyc", "new", "york", "city"}


def route_keys(name):
    """Name variants to search OSM route relations with, most specific first.
    'Delhi Metro Yellow Line' -> ['Delhi Metro Yellow Line', 'Yellow Line', 'Yellow'];
    'Rapid Metro Gurgaon' -> ['Rapid Metro Gurgaon', 'Rapid Metro', 'Rapid']"""
    words = name.replace("'", "").split()
    keys = [name]
    core = [w for w in words if w.lower() not in GENERIC]
    if words and words[-1].lower() == "line" and core:
        keys.append(" ".join(core) + " Line")
    if len(words) >= 2:
        keys.append(" ".join(words[:2]))
    if core:
        keys.append(core[0])
    out = []
    for k in keys:
        if k and k not in out:
            out.append(k)
    return out


def resolve_route(db, ent, resolved_places, offline=False):
    stops = []
    for s in ent.get("stops") or []:
        if s in resolved_places and resolved_places[s]:
            stops.append(resolved_places[s]["label_point"])
        else:
            r = resolve_place(db, {"name_en": s, "kind": "city", "country": ent.get("country"), "visual": "pin"}, offline)
            if r:
                stops.append(r["label_point"])
    centre = stops[0] if stops else ([ent["lon"], ent["lat"]] if ent.get("lat") is not None else None)
    country = ent.get("country")
    if not country and centre:
        c = db.layer("countries")
        hit = c[c.geometry.contains(Point(centre))]
        country = hit.ISO_A2.iloc[0] if len(hit) else None
        if len(hit) and hit.ADM0_A3.iloc[0] == "IND":
            country = "IN"
    geom = None
    if not offline and centre and ent.get("mode") in ("metro", "rail", "road") and ent.get("name_en"):
        try:
            geom = overpass_route(ent, centre, country)
        except Exception as e:
            print(f"[geocode] Overpass failed for {ent.get('name_en')!r}: {e}", file=sys.stderr)
    if geom is not None and not geom.is_empty:
        return pack(geom, "OpenStreetMap Overpass (ODbL)", "overpass", 0.8, ent.get("name_en"),
                    {"stops": stops, "mode": ent.get("mode")})
    if len(stops) >= 2:
        return pack(LineString(stops), "straight line through stops", "approx", 0.3, ent.get("name_en"),
                    {"stops": stops, "mode": ent.get("mode")})
    return None


def context_for(db, resolved):
    """Polygons the renderer needs around the story: the countries touched + their neighbours, and Indian states."""
    pts = [Point(r["label_point"]) for r in resolved.values() if r]
    if not pts:
        return {}
    c = db.layer("countries")
    hit = c[c.geometry.intersects(unary_union(pts).buffer(0.01))]
    near = c[c.geometry.intersects(unary_union(list(hit.geometry)).buffer(2.0))] if len(hit) else hit
    ctx = {"countries": [{"a3": r.ADM0_A3, "name": r.NAME, "geometry": mapping(simplify_for_view(r.geometry))}
                         for _, r in near.iterrows()]}
    if "IND" in set(hit.ADM0_A3):
        a = db.layer("admin1")
        ctx["india_states"] = [{"name": r["name"], "name_hi": r["name_hi"], "geometry": mapping(simplify_for_view(r.geometry))}
                               for _, r in a[a.adm0_a3 == "IND"].iterrows()]
    return ctx


def geocode(entities, offline=False):
    db = GeoDB()
    out, resolved = [], {}
    main = next((x for x in entities["entities"] if x["id"] == entities.get("main_place")), None)
    main_name = main.get("name_en") if main and main.get("kind") in ("city", "district") else None
    for e in entities["entities"]:
        if e["type"] != "place":
            continue
        if e.get("kind") in SMALL_KINDS and main_name and e["id"] != entities.get("main_place"):
            e = dict(e, _context=[main_name])
        r = resolve_place(db, e, offline)
        resolved[e["id"]] = r
        out.append({"id": e["id"], "type": "place", "name_en": e.get("name_en"), "kind": e.get("kind"), "start": e["start"],
                    "resolved": r})
        tag = f"{r['method']} {r['name']} conf {r['confidence']}" if r else "NOT FOUND"
        print(f"[geocode] {e['id']} {e.get('name_en')!r} ({e.get('kind')}): {tag}")
    for e in entities["entities"]:
        if e["type"] != "route":
            continue
        r = resolve_route(db, e, resolved, offline)
        out.append({"id": e["id"], "type": "route", "name_en": e.get("name_en"), "mode": e.get("mode"), "start": e["start"],
                    "resolved": r})
        tag = f"{r['method']} conf {r['confidence']}" if r else "NOT FOUND"
        print(f"[geocode] {e['id']} route {e.get('name_en')!r}: {tag}")
    missing = [x["id"] for x in out if not x["resolved"]]
    # low confidence = an approximate point/line (LLM coordinates, straight line through stops): render it, but show
    # it in the review report so a wrong or crude shape never ships unnoticed
    review = [{"id": x["id"], "name": x["name_en"], "method": x["resolved"]["method"], "confidence": x["resolved"]["confidence"]}
              for x in out if x["resolved"] and x["resolved"]["confidence"] < 0.5]
    for r in review:
        print(f"[geocode] REVIEW {r['id']} {r['name']!r}: only {r['method']} (confidence {r['confidence']})")
    return {"entities_file": entities.get("transcript"), "items": sorted(out, key=lambda x: x["start"]),
            "missing": missing, "needs_review": review, "context": context_for(db, resolved)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entities")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offline", action="store_true", help="never call Nominatim/Overpass")
    a = ap.parse_args()
    res = geocode(read_json(a.entities), a.offline)
    write_json(a.out, res)
    print(f"[geocode] {len(res['items'])} items, {len(res['missing'])} missing -> {a.out}")


if __name__ == "__main__":
    main()
