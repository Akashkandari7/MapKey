"""Stage 4: transcript + entities + geo -> scenes.json (the shot list the renderer draws).

    python -m pipeline.plan --episode work/<ep> [--range 0:10]

scenes.json (validated against SCHEMA below; hand-editable, then re-run from render):
    {
      "version": 1, "duration": 50.9, "fps": 30, "width": 1080, "height": 1920, "look": "metro_dark",
      "audio": "narration.wav",
      "geometry": {"g_e1": {"kind": "polygon"|"line"|"point", "name": "...", "coords": [...]}},   # Web Mercator metres
      "shots": [
        {"id": "s01", "kind": "map", "start": 0, "end": 33.5,
         "camera": [{"t": 0, "x": .., "y": .., "mpp": 60.0, "ease": "easeOutSine"}, ...],   # mpp = metres per pixel
         "drift_per_s": 0.03,
         "focus": "g_e1",                                                  # dim everything outside this polygon
         "layers": [
           {"type": "route", "geom": "g_e2", "color": "red", "in": 0.92, "draw_s": 1.1},
           {"type": "tint", "geom": "g_e9", "in": ..},
           {"type": "pin", "geom": "g_e7", "label": "Guru Dronacharya", "in": 13.4},
           {"type": "title", "text": "Rapid Metro", "style": "title", "x": 540, "y": 600, "in": .., "out": ..},
           {"type": "stat", "x": 300, "y": 900, "align": "center", "out": 7.9,
            "lines": [{"text": "11 Stations", "style": "label", "in": 2.43}, {"text": "11.7 KM", "style": "value", "in": 4.67}]}
         ]},
        {"id": "s02", "kind": "facecam", "start": 33.5, "end": 50.9, "src": null}      # null = placeholder until you give the clip
      ],
      "credits": ["Contains modified Copernicus Sentinel data 2024-2025", "© OpenStreetMap contributors"]
    }

Every rule comes from style/tokens.json (looks.<look>, motion, plan). Times come from word timestamps.
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, box, shape
from shapely.ops import linemerge

from pipeline.common import ROOT, read_json, write_json
from pipeline.geo.proj import geom_to_3857

SCHEMA = {
    "type": "object",
    "required": ["version", "duration", "fps", "width", "height", "look", "geometry", "shots"],
    "properties": {
        "duration": {"type": "number", "minimum": 0},
        "geometry": {"type": "object"},
        "shots": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["id", "kind", "start", "end"],
            "properties": {
                "kind": {"enum": ["map", "facecam", "broll"]},
                "start": {"type": "number"}, "end": {"type": "number"},
                "camera": {"type": "array", "items": {"type": "object", "required": ["t", "x", "y", "mpp"]}},
                "layers": {"type": "array", "items": {"type": "object", "required": ["type"], "properties": {
                    "type": {"enum": ["route", "tint", "pin", "title", "stat"]}}}}}}},
    },
}


# keep in sync with EASE in pipeline/render/mapani.js (the planner predicts where things are on screen)
EASES = {
    "linear": lambda u: u,
    "easeOutSine": lambda u: math.sin(u * math.pi / 2),
    "easeInOutSine": lambda u: -(math.cos(math.pi * u) - 1) / 2,
    "easeOutCubic": lambda u: 1 - (1 - u) ** 3,
    "easeInOutCubic": lambda u: 4 * u ** 3 if u < .5 else 1 - (-2 * u + 2) ** 3 / 2,
}


class Planner:
    def __init__(self, ep, tokens, rng=None):
        self.ep = Path(ep)
        self.T = tokens
        self.tr = read_json(self.ep / "transcript.json")
        self.ents = read_json(self.ep / "entities.json")
        self.geo = read_json(self.ep / "geo.json")
        self.look = tokens["default_look"]
        self.L = tokens["looks"][self.look]
        self.P = tokens["plan"]
        self.W, self.H, self.fps = tokens["format"]["width"], tokens["format"]["height"], tokens["format"]["fps"]
        self.safe = tokens["format"]["safe_area"]
        self.dur = self.tr["duration"]
        self.range = rng or (0.0, self.dur)
        self.geom3857 = {}
        self.out_geom = {}
        by_id = {e["id"]: e for e in self.ents["entities"]}
        self.by_id = by_id
        for it in self.geo["items"]:
            if it["resolved"]:
                self.geom3857[it["id"]] = geom_to_3857(shape(it["resolved"]["geometry"]))

    # ---------- geometry helpers ----------
    def main_id(self):
        return self.ents.get("main_place")

    def route_line(self, g, stops=None):
        """One ordered polyline for drawing a route on: merge OSM pieces, drop the opposite-direction duplicate track,
        chain the rest end-to-end starting near the first stop."""
        if g.geom_type == "LineString":
            return g
        merged = linemerge(g) if g.geom_type == "MultiLineString" else g
        parts = list(getattr(merged, "geoms", [merged]))
        parts.sort(key=lambda p: -p.length)
        kept = []
        for p in parts:  # metro relations list both directions: skip parts lying on top of an already kept part
            if kept and MultiLineString(kept).buffer(40).contains(p.interpolate(0.5, normalized=True)) \
                    and MultiLineString(kept).buffer(40).intersection(p).length > 0.8 * p.length:
                continue
            kept.append(p)
        if len(kept) == 1:
            return kept[0]
        start_pt = Point(stops[0]) if stops else Point(kept[0].coords[0])
        chain, rest = [], kept[:]
        cur = min(rest, key=lambda p: min(Point(p.coords[0]).distance(start_pt), Point(p.coords[-1]).distance(start_pt)))
        rest.remove(cur)
        if Point(cur.coords[-1]).distance(start_pt) < Point(cur.coords[0]).distance(start_pt):
            cur = LineString(cur.coords[::-1])
        chain += list(cur.coords)
        while rest:
            end = Point(chain[-1])
            nxt = min(rest, key=lambda p: min(Point(p.coords[0]).distance(end), Point(p.coords[-1]).distance(end)))
            if min(Point(nxt.coords[0]).distance(end), Point(nxt.coords[-1]).distance(end)) > 3000:
                break  # a far-away fragment is not part of this line's path
            rest.remove(nxt)
            if Point(nxt.coords[-1]).distance(end) < Point(nxt.coords[0]).distance(end):
                nxt = LineString(nxt.coords[::-1])
            chain += list(nxt.coords)
        return LineString(chain)

    def smooth_through(self, pts, n=24):
        """Catmull-Rom through stop points: an 'approx' route (planned line) drawn as a smooth curve, not a zig-zag."""
        if len(pts) < 3:
            return LineString(pts)
        p = [pts[0]] + pts + [pts[-1]]
        out = []
        for i in range(1, len(p) - 2):
            p0, p1, p2, p3 = map(np.array, (p[i - 1], p[i], p[i + 1], p[i + 2]))
            for t in np.linspace(0, 1, n, endpoint=False):
                out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                                  (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
        out.append(np.array(pts[-1]))
        return LineString([tuple(x) for x in out])

    def add_geom(self, gid, g, kind, name, tol):
        g = g.simplify(tol, preserve_topology=True) if tol > 0 else g
        if kind == "point":
            coords = [round(g.x, 1), round(g.y, 1)]
        elif kind == "line":
            coords = [[round(x, 1), round(y, 1)] for x, y in g.coords]
        else:
            polys = list(getattr(g, "geoms", [g]))
            coords = [[[[round(x, 1), round(y, 1)] for x, y in ring.coords] for ring in [p.exterior, *p.interiors]]
                      for p in polys if p.geom_type == "Polygon"]
        self.out_geom[gid] = {"kind": kind, "name": name, "coords": coords}
        return gid

    # ---------- camera ----------
    def fit(self, bounds, fill):
        """Camera (x, y, mpp) that fits bounds into the safe area at `fill` of its size."""
        minx, miny, maxx, maxy = bounds
        sw = self.W - self.safe["left"] - self.safe["right"]
        sh = self.H - self.safe["top"] - self.safe["bottom"]
        bw, bh = max(maxx - minx, 1), max(maxy - miny, 1)
        mpp = max(bw / (sw * fill), bh / (sh * fill))
        # centre of the safe area, in screen px offset from the frame centre
        ox = (self.safe["left"] - self.safe["right"]) / 2
        oy = (self.safe["top"] - self.safe["bottom"]) / 2
        cx, cy = (minx + maxx) / 2 - ox * mpp, (miny + maxy) / 2 + oy * mpp
        return cx, cy, mpp

    def to_screen(self, cam, x, y):
        cx, cy, mpp = cam
        return (x - cx) / mpp + self.W / 2, (cy - y) / mpp + self.H / 2

    def cam_at(self, keys, t, drift):
        """Same math as the renderer (mapani.js camAt): piecewise eased keys, zoom drift on top."""
        if t <= keys[0]["t"]:
            k = keys[0]
            c = (k["x"], k["y"], k["mpp"])
        else:
            c = (keys[-1]["x"], keys[-1]["y"], keys[-1]["mpp"])
            for a, b in zip(keys, keys[1:]):
                if t < b["t"]:
                    t0 = b.get("t0", a["t"])
                    u = 0 if t <= t0 else min(1, (t - t0) / max(1e-6, b["t"] - t0))
                    e = EASES[b.get("ease", "easeOutSine")](u)
                    lm = math.log(a["mpp"]) + (math.log(b["mpp"]) - math.log(a["mpp"])) * e
                    c = (a["x"] + (b["x"] - a["x"]) * e, a["y"] + (b["y"] - a["y"]) * e, math.exp(lm))
                    break
        return c[0], c[1], c[2] * math.exp(-drift * t)

    # ---------- text helpers ----------
    def short_name(self, e):
        """'Gurgaon Rapid Metro' -> 'Rapid Metro' (drop the main place's name, which the viewer already sees)."""
        from pipeline.geocode import CANON
        name = e.get("display") or e.get("name_en") or e.get("quote", "")
        main = self.by_id.get(self.main_id(), {})
        drop = {CANON.get(w.lower(), w.lower()) for w in (main.get("name_en") or "").split()}
        words = name.split()
        kept = [w for w in words if CANON.get(w.lower(), w.lower()) not in drop]
        return " ".join(kept) if kept else name

    def stat_style(self, e):
        unit = (e.get("unit") or "").lower()
        disp = (e.get("display") or "").lower()
        if any(u in unit or disp.endswith(u) for u in self.P["value_units"]):
            return "value"
        return "label"

    # ---------- plan ----------
    def plan(self):
        T, P = self.T, self.P
        lead = T["motion"]["camera"]["lead_before_word"]
        camT = T["motion"]["camera"][self.look]
        drift = camT.get("drift_zoom_per_s", 0)
        sections = self.ents.get("sections", [])
        face_start = next((s["start"] for s in sections if s["role"] in ("opinion", "cta")), self.dur)
        a, b = self.range
        map_end = min(face_start, b)

        main = self.main_id()
        main_g = self.geom3857.get(main)
        subjects = []
        for e in self.ents["entities"]:
            if e["type"] in ("place", "route") and e["id"] in self.geom3857 and e["start"] < map_end:
                subjects.append(e)
        subjects.sort(key=lambda e: e["start"])

        layers, keys = [], []
        # first view: the main place, a little wider than its fit, camera already moving from frame 0
        first = main_g if main_g is not None else self.geom3857[subjects[0]["id"]]
        cx, cy, mpp = self.fit(first.bounds, P["fit"]["main_place"])
        keys.append({"t": a, "x": cx, "y": cy, "mpp": mpp * P["first_view_zoom_out"]})
        focus = None
        if main_g is not None and main_g.geom_type in ("Polygon", "MultiPolygon") and self.L["basemap"]["focus_dim"]["enabled"]:
            focus = self.add_geom("g_" + main, main_g, "polygon", self.by_id[main].get("name_en"), mpp / 3)

        clip_box = None
        if main_g is not None:
            mx0, my0, mx1, my1 = main_g.bounds
            k = P["route_clip_to_main"]
            cxm, cym, hw, hh = (mx0 + mx1) / 2, (my0 + my1) / 2, (mx1 - mx0) / 2 * k, (my1 - my0) / 2 * k
            clip_box = box(cxm - hw, cym - hh, cxm + hw, cym + hh)

        colors = list(P["route_colors"])
        used = {}
        hook_end = next((s["end"] for s in sections if s["role"] == "hook"), 0)
        title_done = False
        for e in subjects:
            g = self.geom3857[e["id"]]
            if e["type"] == "route":
                stops = [list(self.geom3857[s].representative_point().coords[0]) for s in (e.get("stops") or [])
                         if s in self.geom3857]
                res = next(it["resolved"] for it in self.geo["items"] if it["id"] == e["id"])
                line = self.smooth_through(stops) if res["method"] == "approx" and len(stops) >= 3 else self.route_line(g, stops)
                # far-reaching lines (Yellow Line runs 50 km into Delhi) are shown where the story is
                view_line = line.intersection(clip_box) if clip_box is not None and not clip_box.contains(line) else line
                if view_line.is_empty:
                    view_line = line
                name_l = (e.get("name_en") or "").lower()
                color = next((c for k, c in P["named_route_colors"].items() if re.search(rf"\b{k}\b", name_l)), None)
                if color is None:
                    color = next(c for c in colors if c not in used.values()) if len(used) < len(colors) else colors[0]
                used[e["id"]] = color
                gid = self.add_geom("g_" + e["id"], line, "line", e.get("name_en"), mpp / 4)
                layers.append({"type": "route", "geom": gid, "color": color, "in": e["start"],
                               "draw_s": T["motion"]["route_draw"]["duration"], "ease": T["motion"]["route_draw"]["ease"]})
                target = view_line.bounds
                fill = P["fit"]["route"]
            else:
                kind = e.get("kind")
                if e["id"] == main:
                    continue
                if g.geom_type == "Point" or e.get("visual") in ("pin", "label"):
                    gid = self.add_geom("g_" + e["id"], g.representative_point() if g.geom_type != "Point" else g, "point",
                                        e.get("name_en"), 0)
                    label = (e.get("display") or e.get("name_en") or "").replace(" Metro Station", "").replace(" Gurugram", "")
                    layers.append({"type": "pin", "geom": gid, "label": label,
                                   "in": e["start"] + T["motion"]["label_in"]["start_after_word"]})
                    continue  # pins don't move the camera; it keeps drifting over the area being talked about
                gid = self.add_geom("g_" + e["id"], g, "polygon", e.get("name_en"), mpp / 3)
                layers.append({"type": "tint", "geom": gid, "in": e["start"]})
                target, fill = g.bounds, P["fit"]["place"]
            t_move = max(a, e["start"] - lead)
            cx, cy, mpp_t = self.fit(target, fill)
            # don't zoom out to a far wider view just because a route is long: keep zooming in or holding scale
            mpp_t = min(mpp_t, keys[-1]["mpp"]) if e["type"] == "route" else mpp_t
            dur = float(np.clip(camT["move_s"][0] + 0.4 * (camT["move_s"][1] - camT["move_s"][0]), *camT["move_s"]))
            keys.append({"t0": t_move, "t": t_move + dur, "x": cx, "y": cy, "mpp": mpp_t, "ease": camT["ease"]})
            # hook title: the first subject named in the hook gets a title on the map, above it
            if not title_done and e["start"] <= hook_end:
                cam = self.cam_at(keys, e["start"] + 0.3, drift)
                bx = view_line.bounds if e["type"] == "route" else g.bounds
                sx, sy = self.to_screen(cam, (bx[0] + bx[2]) / 2, bx[3])
                y = max(self.safe["top"] + 60, sy - P["title_above_subject"] * self.H)
                layers.append({"type": "title", "text": self.short_name(e), "style": "title_on_map",
                               "x": round(float(np.clip(sx, 260, self.W - 260)), 1), "y": round(y, 1),
                               "in": e["start"] + T["motion"]["text_in"]["start_after_word"]})
                title_done = True
        # keys must be increasing; overlapping moves: a later move takes over from wherever the camera is
        keys.sort(key=lambda k: k["t"])
        for i in range(1, len(keys)):
            keys[i]["t0"] = max(keys[i].get("t0", keys[i - 1]["t"]), keys[i - 1].get("t0", keys[i - 1]["t"]))

        # stats: numbers stacked per subject, screen-fixed beside the subject
        nums = [e for e in self.ents["entities"] if e["type"] == "number" and a <= e["start"] < map_end]
        groups = []
        for n in nums:
            if groups and groups[-1]["ref"] == n.get("refers_to") and n["start"] - groups[-1]["lines"][-1]["start"] < P["stack_window_s"]:
                groups[-1]["lines"].append(n)
            else:
                groups.append({"ref": n.get("refers_to"), "lines": [n]})
        subj_times = [e["start"] for e in subjects] + [map_end]
        for gi, grp in enumerate(groups):
            t_in = grp["lines"][0]["start"]
            t_out = min([t for t in subj_times if t > grp["lines"][-1]["start"] + 0.5] + [map_end])
            cam = self.cam_at(keys, t_in + 1.0, drift)
            ref_g = self.geom3857.get(grp["ref"]) or main_g
            if grp["ref"] in self.geom3857 and self.by_id.get(grp["ref"], {}).get("type") == "route" and clip_box is not None:
                ref_g = ref_g.intersection(clip_box) if not ref_g.intersection(clip_box).is_empty else ref_g
            bx = ref_g.bounds
            x0, y0 = self.to_screen(cam, bx[0], bx[3])
            x1, y1 = self.to_screen(cam, bx[2], bx[1])
            left_space, right_space = x0 - self.safe["left"], (self.W - self.safe["right"]) - x1
            m = P["stat_side_margin"] * self.W
            if left_space >= right_space:
                x = (self.safe["left"] + max(self.safe["left"] + 1, x0 - m)) / 2
            else:
                x = (min(self.W - self.safe["right"] - 1, x1 + m) + self.W - self.safe["right"]) / 2
            x = float(np.clip(x, self.safe["left"] + 170, self.W - self.safe["right"] - 170))
            y = float(np.clip((y0 + y1) / 2, self.safe["top"] + 200, self.H - self.safe["bottom"] - 200))
            layers.append({"type": "stat", "x": round(x, 1), "y": round(y, 1), "align": "center", "out": round(t_out, 3),
                           "lines": [{"text": n.get("display") or n["quote"], "style": self.stat_style(n),
                                      "in": round(n["start"] + T["motion"]["number_in"]["start_after_word"], 3)}
                                     for n in grp["lines"]]})
        # titles end where the first stat begins (r1: "Rapid Metro" gives way to "11 Stations")
        first_stat = min([l["lines"][0]["in"] for l in layers if l["type"] == "stat"] + [map_end])
        for l in layers:
            if l["type"] == "title":
                l["out"] = round(max(l["in"] + 1.2, first_stat), 3)

        shots = [{"id": "s01", "kind": "map", "start": a, "end": map_end, "camera": [
                    {k: (round(v, 2) if isinstance(v, float) else v) for k, v in key.items()} for key in keys],
                  "drift_per_s": drift, "focus": focus, "layers": layers}]
        if map_end < b:
            shots.append({"id": "s02", "kind": "facecam", "start": map_end, "end": b, "src": None})
        credits = sorted({c for c in [self.L["basemap"].get("credit")] if c})
        return {"version": 1, "duration": round(b - a, 3), "offset": a, "fps": self.fps, "width": self.W, "height": self.H,
                "look": self.look, "audio": next((p.name for p in self.ep.glob("narration.*")), None),
                "geometry": self.out_geom, "shots": shots, "credits": credits}


def validate(scenes):
    import jsonschema
    jsonschema.validate(scenes, SCHEMA)
    shots = scenes["shots"]
    t = scenes.get("offset", 0)
    for s in shots:  # shots must tile the timeline with no gaps or overlaps
        if abs(s["start"] - t) > 1e-3:
            raise ValueError(f"gap/overlap before shot {s['id']}: expected start {t}, got {s['start']}")
        t = s["end"]
    if abs(t - (scenes.get("offset", 0) + scenes["duration"])) > 1e-3:
        raise ValueError(f"shots end at {t}, audio ends at {scenes['duration']}")
    for s in shots:
        for l in s.get("layers", []):
            if "geom" in l and l["geom"] not in scenes["geometry"]:
                raise ValueError(f"layer refers to missing geometry {l['geom']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="work/<episode> folder")
    ap.add_argument("--range", help="plan only a:b seconds (tests)")
    ap.add_argument("--out")
    a = ap.parse_args()
    rng = tuple(float(v) for v in a.range.split(":")) if a.range else None
    tokens = read_json(ROOT / "style/tokens.json")
    sc = Planner(a.episode, tokens, rng).plan()
    validate(sc)
    out = Path(a.out) if a.out else Path(a.episode) / "scenes.json"
    write_json(out, sc)
    n = sum(len(s.get("layers", [])) for s in sc["shots"])
    print(f"[plan] {len(sc['shots'])} shots, {n} layers, {len(sc['geometry'])} geometries -> {out}")


if __name__ == "__main__":
    main()
