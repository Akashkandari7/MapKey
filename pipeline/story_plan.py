"""Story mode, stage 4: storyboard.json + visuals.json + geo.json -> scenes.json (mode "story") for story.js.

    python -m pipeline.story_plan --episode work/<ep> [--handle @name]

One shot per beat. A shot runs from its beat's first word to the next beat's first word (the first from 0, the last to
the end of the audio); the transition into it is centred on that cut. Photos get a slow move, news cards their
marker sweeps on the spoken word, maps a camera that dives from wide to the subject with the places coloured,
labelled and circled on their words. Every style value comes from style/tokens.json ("story" and looks.story_map).
"""
import argparse
import math
import re
from pathlib import Path

from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import substring, unary_union

from pipeline.common import ROOT, read_json, require_env, sha, write_json
from pipeline.plan import Planner, geo_layer, label_point, validate
from pipeline.geo.proj import geom_to_3857

MOTIONS = {  # image moves: scale from->to, offset from->to (px)
    "push_in": {"scale": [1.0, 1.14]},
    "pull_out": {"scale": [1.15, 1.0]},
    "pan_left": {"scale": [1.16, 1.16], "from": [70, 0], "to": [-70, 0]},
    "pan_right": {"scale": [1.16, 1.16], "from": [-70, 0], "to": [70, 0]},
    "rise": {"scale": [1.16, 1.18], "from": [0, 90], "to": [0, -90]},
    "sink": {"scale": [1.18, 1.16], "from": [0, -90], "to": [0, 90]},
}
WORLD_MPP = 16000  # 'world' start: half the globe across, the whole height of the Mercator world on screen


def parse_count(text):
    """'80%' -> {from 0, to 80, suffix %}; '₹50,000' -> prefix ₹, to 50000; '2.5 lakh' -> to 2.5, suffix ' lakh'."""
    m = re.match(r"^(\D*?)(\d[\d,]*(?:\.\d+)?)(.*)$", text.strip())
    if not m:
        return None
    num = m.group(2).replace(",", "")
    dec = len(num.split(".")[1]) if "." in num else 0
    return {"from": 0, "to": float(num), "decimals": dec, "prefix": m.group(1), "suffix": m.group(3),
            "sep": "," if "," in m.group(2) or float(num) >= 10000 else ""}


class StoryPlanner:
    def __init__(self, ep, tokens, handle=None):
        self.ep = Path(ep)
        self.T = tokens
        self.ST = tokens["story"]
        self.P = Planner(ep, tokens)  # geometry, camera fitting, India clipping
        self.P.look = "story_map"
        self.P.L = tokens["looks"]["story_map"]
        self.sb = read_json(self.ep / "storyboard.json")
        self.vis = read_json(self.ep / "visuals.json") if (self.ep / "visuals.json").exists() else {}
        self.words = self.P.tr["words"]
        self.dur = self.P.tr["duration"]
        self.W, self.H = self.P.W, self.P.H
        self.handle = handle

    # ---------- text ----------
    def text_layer(self, b, end):
        t = b.get("text")
        if not t or t.get("preset") in (None, "none"):
            return None
        pre = self.ST["text"]["presets"].get(t["preset"])
        if not pre:
            return None
        roles = list(pre["roles"].keys())
        lines = []
        for ln in t["lines"]:
            role = ln.get("role") if ln.get("role") in pre["roles"] else roles[min(len(lines), len(roles) - 1)]
            if "color" in pre["roles"][role] and "fill" not in pre["roles"][role]:
                role = roles[0]  # a colour-only role (question 'hl') is for runs, not whole lines
            L = {"text": ln["text"], "role": role, "in": round(ln.get("t", b["start"]) + 0.05, 3)}
            if t["preset"] == "stat" and role == "num" or ln.get("count_to") is not None:
                c = parse_count(ln["text"])
                if c:
                    L["count"] = c
            lines.append(L)
        if not lines:
            return None
        lines.sort(key=lambda x: x["in"])
        # typed body lines (stat sentences) read as a sentence, not shouted: caps -> sentence case, and a long one is
        # wrapped into lines that type in one after another
        out = []
        for L in lines:
            r = pre["roles"][L["role"]]
            if not r.get("type") or t["preset"] == "caption_box":
                out.append(L)
                continue
            txt = L["text"]
            if txt.isupper() and len(txt) > 3:
                keep = {"UP", "UK", "US", "USA", "RBI", "GDP", "NCR", "CIA", "UN", "EU", "AI"}
                txt = " ".join(w if w.strip(".,!?") in keep else w.lower() for w in txt.split())
                if not out:  # the sentence starts here (before the number): capital first letter
                    txt = txt[0].upper() + txt[1:]
            maxc = max(12, int(0.84 * self.W / (r["size"] * 0.55)))
            rows, cur = [], ""
            for w in txt.split():
                if cur and len(cur) + 1 + len(w) > maxc:
                    rows.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            rows.append(cur)
            for k, row in enumerate(rows):
                out.append({**L, "text": row, "in": round(L["in"] + 0.3 * k, 3)})
        lines = out
        # the block appears with its first line; lines that would type in after the shot has gone are moved earlier
        for L in lines:
            L["in"] = min(L["in"], end - 0.6)
        layer = {"type": "text", "preset": t["preset"], "in": lines[0]["in"], "out": round(end - 0.15, 3), "lines": lines}
        if t["preset"] == "caption_box":  # the sentence types in word by word over the time it is spoken
            ws = lines[0]["text"].split()
            t0, t1 = lines[0]["in"], max(lines[0]["in"] + 0.6, end - 0.5)
            lines[0]["words"] = [{"w": w, "t": round(t0 + (t1 - t0) * i / max(1, len(ws)), 3)} for i, w in enumerate(ws)]
        return layer

    # ---------- maps ----------
    def country_geom(self, ent):
        cc = (ent.get("country") or "").upper()
        if not cc:
            return None
        c = geo_layer("countries")
        hit = c[c.ISO_A2 == cc]
        return geom_to_3857(hit.geometry.iloc[0]) if len(hit) else None

    @staticmethod
    def main_part(g):
        """For framing: a country's main land, without far-away islands / overseas territories (France's
        Natural Earth shape reaches French Guiana and Reunion: framing all of it zoomed out to the whole planet)."""
        if g is None or g.geom_type != "MultiPolygon":
            return g
        parts = sorted(g.geoms, key=lambda q: -q.area)
        big = parts[0]
        r = math.sqrt(big.area)
        return unary_union([q for q in parts if q.distance(big) < 1.5 * r])

    def river_line(self, eid):
        """A river as a line: Nominatim often answers with the riverbank area (a polygon) or nothing; the offline
        Natural Earth rivers layer has the centre line."""
        e = self.P.by_id.get(eid, {})
        name = re.sub(r"\s+(River|Nadi)$", "", e.get("name_en") or "", flags=re.I).strip().lower()
        if not name:
            return None
        r = geo_layer("rivers")
        hit = r[(r.name.fillna("").str.lower() == name) | (r.name_en.fillna("").str.lower() == name)]
        if not len(hit):
            return None
        from shapely.ops import linemerge
        g = geom_to_3857(unary_union(list(hit.geometry)))
        g = linemerge(g) if g.geom_type == "MultiLineString" else g
        return max(g.geoms, key=lambda q: q.length) if g.geom_type == "MultiLineString" else g

    def map_shot(self, b, a, z, trans_in, v=None):
        P, m = self.P, b["map"]
        ents = P.by_id
        places = [p for p in m.get("places", []) if p in P.geom3857]
        focus = m.get("focus") if m.get("focus") in P.geom3857 else (places[0] if places else None)
        if focus is None:
            return None
        fg = P.geom3857[focus]
        for p in places:  # rivers are drawn as flowing lines, whatever shape the geocoder found
            if ents.get(p, {}).get("kind") == "river" and P.geom3857[p].geom_type not in ("LineString", "MultiLineString"):
                ln = self.river_line(p)
                if ln is not None:
                    P.geom3857[p] = ln
        if focus not in places:  # the place the camera ends on is always coloured too
            places = [focus] + places
        polys = [p for p in places if P.geom3857[p].geom_type in ("Polygon", "MultiPolygon")]
        lines = [p for p in places if P.geom3857[p].geom_type in ("LineString", "MultiLineString")]
        points = [p for p in places if P.geom3857[p].geom_type == "Point"]
        # end framing: the focus and the coloured places near it (a far one would pull the camera out)
        fm = self.main_part(fg)
        tgt = [fm] + [self.main_part(P.geom3857[p]) for p in polys + lines
                      if P.geom3857[p].distance(fm) < 4 * max(1.0, math.sqrt(max(fm.area, 1)))]
        tb = list(unary_union(tgt).bounds)
        minsz = 40000  # a city point still gets ~40 km of context
        if tb[2] - tb[0] < minsz:
            cx_ = (tb[0] + tb[2]) / 2
            tb[0], tb[2] = cx_ - minsz / 2, cx_ + minsz / 2
        if tb[3] - tb[1] < minsz:
            cy_ = (tb[1] + tb[3]) / 2
            tb[1], tb[3] = cy_ - minsz / 2, cy_ + minsz / 2
        cx, cy, mpp = P.fit(tb, 0.62)
        mpp = min(mpp, WORLD_MPP)  # never wider than the world view (the Mercator world ends top and bottom)
        if m.get("style") == "photo":  # the picture inside the outline needs the place big
            cx, cy, mpp = P.fit(fg.bounds, 0.8)
        elif m.get("style") == "flow":  # room around the place for the rivers / routes coming in and their boxes
            fb = fg.bounds
            r = max(60000.0, 2.2 * max(fb[2] - fb[0], fb[3] - fb[1]))
            cx, cy, mpp = P.fit((fb[0] - r, fb[1] - r, fb[2] + r, fb[3] + r), 0.95)
        sw = m.get("start_wide") or "none"
        # a magnifier only on a spot: not on the area the camera frames, not on a big region (it shows a flat blur)
        def spot(c):
            g = P.geom3857.get(c.get("place"))
            if g is None or (c.get("place") == focus and fg.geom_type != "Point"):
                return False
            b0 = g.bounds
            return g.geom_type == "Point" or max(b0[2] - b0[0], b0[3] - b0[1]) < 0.25 * max(tb[2] - tb[0], tb[3] - tb[1])
        m["callouts"] = [c for c in m.get("callouts", []) if spot(c)]
        circled_ids = {c.get("place") for c in m["callouts"]}
        country = self.country_geom(ents.get(focus, {}))
        if (m.get("style") == "heat") and country is not None:  # a heat map is read across the whole country
            cx, cy, mpp = P.fit(self.main_part(country).bounds, 0.86)
            sw = "none"
        if sw == "world":
            x0, y0, m0 = cx, max(-1.4e7, min(1.4e7, cy)), WORLD_MPP
        elif sw == "continent":
            x0, y0, m0 = cx, cy, min(WORLD_MPP, mpp * 10)
        elif sw == "country" and country is not None:
            x0, y0, m0 = P.fit(self.main_part(country).bounds, 0.8)
            m0 = max(m0, mpp * 1.6)
        else:
            x0, y0, m0 = cx, cy, mpp * 1.7
        dur = z - a
        # from far out, the wide view holds a moment (the viewer reads where we are) before the dive
        hold = 0.8 if sw != "none" and dur >= 3.2 else 0.05
        move = min(3.6, max(1.4, 0.62 * dur - hold))
        keys = [{"t": round(a - trans_in / 2, 3), "x": round(x0, 1), "y": round(y0, 1), "mpp": round(m0, 3)},
                {"t0": round(a + hold, 3), "t": round(a + hold + move, 3), "x": round(cx, 1), "y": round(cy, 1),
                 "mpp": round(mpp, 3), "ease": "easeInOutCubic"}]
        move += hold
        # places named one after another far apart on the globe ('France, Chile, Spain, India'): the camera whips to
        # each on its name instead of one view of everything (that view is wider than the world)
        spread = [p for p in polys if ents.get(p, {}).get("start") is not None and a <= ents[p]["start"] < z]
        if (m.get("style") or "fill") in ("fill", "districts") and len(spread) >= 2:
            ub = unary_union([self.main_part(P.geom3857[p]) for p in spread]).bounds
            if max(ub[2] - ub[0], ub[3] - ub[1]) > 6.0e6:
                spread.sort(key=lambda p: ents[p]["start"])
                keys = [keys[0]]
                for p in spread:
                    px, py, pm = P.fit(self.main_part(P.geom3857[p]).bounds, 0.42)
                    pm = min(max(pm, 2500.0), WORLD_MPP)
                    t0 = max(keys[-1].get("t", a) + 0.05, ents[p]["start"] - 0.25)
                    keys.append({"t0": round(t0, 3), "t": round(t0 + 0.7, 3), "x": round(px, 1), "y": round(py, 1),
                                 "mpp": round(pm, 3), "ease": "easeInOutCubic"})
                cx, cy, mpp = keys[-1]["x"], keys[-1]["y"], keys[-1]["mpp"]
                move = keys[1]["t"] - a
        tol = mpp / 3
        S = self.ST["map"]
        layers = []
        t_of = lambda eid: max(a + 0.25, min(z - 0.8, ents[eid]["start"])) if ents.get(eid, {}).get("start") is not None and a <= ents[eid]["start"] < z else a + 0.35  # noqa: E731
        style = m.get("style") or "fill"
        gid = lambda eid: f"g_{b['id']}_{eid}"  # noqa: E731
        if style == "heat":
            base = country if country is not None else fg
            g = P.add_geom(gid("heat"), base, "polygon", "heat", tol)
            hx, hy = label_point(fg).coords[0] if fg.geom_type != "Point" else fg.coords[0]
            bx = self.main_part(base).bounds
            layers.append({"type": "region", "style": "heat", "geom": g, "in": round(a + 0.2, 3), "in_s": 1.2,
                           "hot": [round(hx, 1), round(hy, 1)], "radius_m": round(0.55 * max(bx[2] - bx[0], bx[3] - bx[1]), 1)})
        else:
            # the subject yellow (red when it sits among other coloured places), the others in the district colours
            cols = list(S["district_colors"])[1:] + list(S["district_colors"])[:1]
            own = "#F2D21B" if len(polys) == 1 else "#E8202A"
            k = 0
            for p in polys:
                col = own if p == focus and style != "districts" else cols[k % len(cols)]
                if not (p == focus and style != "districts"):
                    k += 1
                layers.append({"type": "region", "style": "glow", "geom": P.add_geom(gid(p), P.geom3857[p], "polygon", ents[p].get("name_en"), tol),
                               "color": col, "in": round(t_of(p), 3)})
        if style == "photo" and (v or {}).get("src"):
            for L in layers:
                if L["type"] == "region" and L["geom"] == gid(focus):
                    L.update({"style": "photo", "src": v["src"]})
        if style == "flow":
            self.flow_layers(b, m, focus, fg, layers, (cx, cy, mpp), a, z, gid, tol)
            lines = []  # drawn as flows
        for p in lines:
            g = P.geom3857[p]
            ln = P.route_line(g) if isinstance(g, MultiLineString) else g
            if ln.geom_type != "LineString":
                continue
            # rivers / canals are long: keep the part in view
            view = box(cx - self.W * mpp, cy - self.H * mpp, cx + self.W * mpp, cy + self.H * mpp)
            part = ln.intersection(view)
            if part.is_empty:
                continue
            if part.geom_type == "MultiLineString":
                part = max(part.geoms, key=lambda q: q.length)
            if part.geom_type != "LineString":
                continue
            layers.append({"type": "route", "geom": P.add_geom(gid(p), part, "line", ents[p].get("name_en"), tol / 2),
                           "color": "#46B8FF", "width": 11, "in": round(t_of(p), 3), "draw_s": 1.6, "ease": "easeInOutSine"})
        # labels: cities as black tags; the focus area's name in big caps on it
        for p in [q for q in points + [q for q in polys if q != focus] if q not in circled_ids]:
            g = P.geom3857[p]
            pt = g if g.geom_type == "Point" else label_point(g)
            layers.append({"type": "tag", "label": ents[p].get("display") or ents[p].get("name_en"), "X": round(pt.x, 1),
                           "Y": round(pt.y, 1), "in": round(t_of(p) + 0.1, 3)})
        circled = any(c.get("place") == focus for c in m.get("callouts", []))
        if fg.geom_type in ("Polygon", "MultiPolygon") and not circled:
            pt = label_point(fg)
            name = ents[focus].get("display") or ents[focus].get("name_en")
            layers.append({"type": "region_label", "text": name, "X": round(pt.x, 1), "Y": round(pt.y, 1),
                           "in": round(max(t_of(focus), a + move * 0.6), 3), "style": "3d" if m.get("tilt") else "flat",
                           "mpp_in": round(mpp, 3)})
        elif focus not in points and not circled:
            layers.append({"type": "tag", "label": ents[focus].get("display") or ents[focus].get("name_en"),
                           "X": round(fg.x, 1), "Y": round(fg.y, 1), "in": round(t_of(focus) + 0.1, 3)})
        for c in m.get("callouts", []):
            if c.get("place") not in P.geom3857:
                continue
            g = P.geom3857[c["place"]]
            pt = g if g.geom_type == "Point" else label_point(g)
            layers.append({"type": "callout", "label": c["label"], "X": round(pt.x, 1), "Y": round(pt.y, 1),
                           "in": round(max(a + move * 0.7, self.words[c["w"]]["start"]), 3)})
        # labels whose spot is off screen (or in the margins) once the camera has settled are dropped
        def on_screen(L):
            if "X" not in L:
                return True
            x, y = P.to_screen((cx, cy, mpp), L["X"], L["Y"])
            return 90 < x < self.W - 90 and 260 < y < self.H - 380
        layers = [L for L in layers if L["type"] not in ("tag", "region_label", "callout") or on_screen(L)]
        # never still: after the dive the camera keeps pushing in (2026-10-22 user: our Delhi map 'stuck', the
        # reference's kept moving)
        shot = {"kind": "map", "camera": keys, "drift_per_s": self.ST["map"]["drift_per_s"], "layers": layers, "focus": None}
        if fg.geom_type in ("Polygon", "MultiPolygon") and style != "heat":
            shot["focus"] = P.add_geom(gid("focus"), fg, "polygon", ents[focus].get("name_en"), tol)
        # 3D tilt only for a city / district / valley: a whole country (India) seen in perspective looks bent (user)
        fb = fg.bounds
        small = max(fb[2] - fb[0], fb[3] - fb[1]) < self.ST["map"]["tilt"]["max_area_m"] and ents[focus].get("kind") != "country"
        if m.get("tilt") and small and style not in ("heat", "photo") and sw not in ("world", "continent", "country"):
            shot["tilt"] = {"rx": self.ST["map"]["tilt"]["rx"]}
        if a < 0.1 and sw != "none":  # the reel opens on this map: fall through clouds into it (q2, q4)
            shot["intro"] = "clouds"
        return shot

    def flow_layers(self, b, m, focus, fg, layers, cam, a, z, gid, tol):
        """Rivers, canals, routes (or a straight-in arc from a named place) flowing INTO the focus, each ending where
        it enters the place, with an info box of its share on the word that says it."""
        P, ents = self.P, self.P.by_id
        cx, cy, mpp = cam
        k_ = 0.4 if m.get("tilt") else 0.58  # a tilted map shows only its middle on screen
        view = box(cx - self.W * mpp * k_, cy - self.H * mpp * k_, cx + self.W * mpp * k_, cy + self.H * mpp * k_)
        centre = fg.centroid if fg.geom_type != "Point" else fg
        for k, f in enumerate(m.get("flows", [])):
            src = f.get("from")
            g = P.geom3857.get(src)
            if g is not None and ents.get(src, {}).get("kind") == "river" and g.geom_type not in ("LineString", "MultiLineString"):
                g = self.river_line(src) or g
            if g is None and ents.get(src, {}).get("lat") is not None:
                # a canal / route the geocoder could not draw ('Upper Ganga Canal'): an arc from its rough location
                g = geom_to_3857(Point(ents[src]["lon"], ents[src]["lat"]))
            if g is None:
                continue
            if g.geom_type in ("LineString", "MultiLineString"):
                ln = P.route_line(g) if g.geom_type == "MultiLineString" else g
                part = ln.intersection(view)
                if part.is_empty:
                    continue
                if part.geom_type == "MultiLineString":
                    part = min(part.geoms, key=lambda q: q.distance(centre))
                if part.geom_type != "LineString" or part.length < 1000:
                    continue
                d = part.project(centre)
                up, down = substring(part, 0, d), substring(part, d, part.length)
                seg = up if up.length >= down.length else LineString(list(down.coords)[::-1])
            else:  # a place: a gentle arc from it (pulled inside the view) to the focus
                p0 = label_point(g) if g.geom_type != "Point" else g
                if not view.contains(p0):
                    t_ = 0.82
                    p0 = Point(centre.x + (p0.x - centre.x) * t_, centre.y + (p0.y - centre.y) * t_)
                    while not view.contains(p0) and t_ > 0.2:
                        t_ -= 0.08
                        p0 = Point(centre.x + (p0.x - centre.x) * 0.9, centre.y + (p0.y - centre.y) * 0.9)
                mx, my = (p0.x + centre.x) / 2, (p0.y + centre.y) / 2
                nx, ny = -(centre.y - p0.y) * 0.22, (centre.x - p0.x) * 0.22
                pts = [((1 - u) ** 2 * p0.x + 2 * (1 - u) * u * (mx + nx) + u * u * centre.x,
                        (1 - u) ** 2 * p0.y + 2 * (1 - u) * u * (my + ny) + u * u * centre.y) for u in [i / 30 for i in range(31)]]
                seg = LineString(pts)
            # stop where it enters the place: the arrow points INTO it
            if fg.geom_type != "Point":
                cut = next((seg.project(Point(q)) for q in seg.coords if fg.contains(Point(q))), None)
                if cut and cut > 0.2 * seg.length:
                    seg = substring(seg, 0, cut)
            t_w = self.words[f["w"]]["start"] if f.get("w") is not None else a + 0.4 + 0.8 * k
            t_in = max(a + 0.3, min(z - 1.2, t_w - 0.2))
            layers.append({"type": "flow", "geom": P.add_geom(gid(f"flow{k}"), seg, "line", src, tol / 2),
                           "in": round(t_in, 3), "draw_s": 1.8})
            q = seg.interpolate(0.35, normalized=True)
            layers.append({"type": "infobox", "label": f["label"], "sub": f.get("sub"), "X": round(q.x, 1), "Y": round(q.y, 1),
                           "dx": -150 if k % 2 == 0 else 150, "dy": -90 if k % 2 == 0 else 120, "in": round(t_in + 0.6, 3)})

    def facecam_punches(self, a, z, i):
        """Face-cam shots alternate wide / closer from one to the next, and a longer one punches in once on a phrase
        start near its middle (a word after a small pause), so the talking head never sits still."""
        F = self.T["facecam"]["story_punch"]
        base = F["scales"][i % len(F["scales"])]
        out = [{"t": 0.0, "scale": base}]
        if z - a >= F["min_shot_s"]:
            mid = a + (z - a) / 2
            ws = [w for k, w in enumerate(self.words) if a + 1.2 < w["start"] < z - 1.2 and k and
                  w["start"] - self.words[k - 1]["end"] > 0.06]
            ws = ws or [w for w in self.words if a + 1.2 < w["start"] < z - 1.2]
            if ws:
                w = min(ws, key=lambda w: abs(w["start"] - mid))
                out.append({"t": round(w["start"] - a, 3), "scale": round(base + F["step"], 3)})
        return out

    # ---------- CTA ----------
    def cta_words(self, b, a, z):
        """The closing question in English, typed on its time, key words in red caps (one cached Gemini call)."""
        said = " ".join(w["w"] for w in self.words[b["w0"]:b["w1"] + 1])
        if b.get("outro"):  # no spoken question (the narration ends on a statement): one written for the topic
            said = "TOPIC: " + (self.sb.get("topic") or "") + " | LAST LINE: " + " ".join(w["w"] for w in self.words[-25:])
        cache = self.ep / "cache" / f"cta_{sha(said, 2)}.txt"
        if cache.exists():
            text = cache.read_text(encoding="utf-8")
        else:
            from google import genai
            client = genai.Client(api_key=require_env("GEMINI_API_KEY", "put GEMINI_API_KEY=... in .env"))
            r = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[("Write the closing comment question for an Instagram explainer reel with this topic and last "
                           "line: one short, natural English sentence in quotes, addressed to the viewer, starting with "
                           "'Friends,', asking their opinion in the comments. " if b.get("outro") else
                           "Translate this closing question of an Instagram reel (Hindi/Hinglish) into one short, natural "
                           "English sentence in quotes, addressed to the viewer, starting with 'Friends,'. ") +
                          "Write its 2-4 key words (the topic) in CAPITALS, everything else normal case. Answer with the "
                          "sentence only.", said])
            text = r.text.strip()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        ws = text.split()
        t0, t1 = a + 0.5, max(a + 1.0, z - 0.7)
        out = []
        for i, w in enumerate(ws):
            core = re.sub(r"[^A-Za-z]", "", w)
            out.append({"w": w, "t": round(t0 + (t1 - t0) * i / max(1, len(ws)), 3),
                        "hl": len(core) >= 2 and core.isupper()})
        return out

    # ---------- plan ----------
    def plan(self):
        beats = self.sb["beats"]
        ST = self.ST
        starts = [0.0] + [b["start"] for b in beats[1:]]
        ends = starts[1:] + [self.dur]
        shots = []
        motions = ["push_in", "pull_out"]
        for i, b in enumerate(beats):
            a, z = round(starts[i], 3), round(ends[i], 3)
            tr = b.get("transition") or ST["transition"]["default"]
            trans = None
            if i > 0 and tr != "cut":
                d = ST["transition"]["dur"].get(tr, 0.4)
                d = min(d, 0.8 * (z - a), 0.8 * (ends[i - 1] - starts[i - 1]))
                trans = {"type": tr, "dur": round(d, 3)}
            v = self.vis.get(b["id"], {})
            kind = b["visual"]
            s = {"id": f"s{i + 1:02d}", "beat": b["id"], "note": b.get("note", ""), "start": a, "end": z}
            if kind == "map":
                ms = self.map_shot(b, a, z, trans["dur"] if trans else 0, v)
                if ms:
                    s.update(ms)
                else:
                    kind = "image"
            if kind == "image":
                s.update({"kind": "image", "src": v.get("src"),
                          "motion": MOTIONS.get(b.get("motion") or motions[i % 2], MOTIONS["push_in"])})
            elif kind == "news":
                s.update({"kind": "news", "src": v.get("src"), "url": v.get("url")})
                t_hl = self.words[b["news"]["w"]]["start"] if b.get("news") else a + 0.8
                hls, k = [], 0
                for ph, rects in (v.get("boxes") or {}).items():
                    if rects:
                        hls.append({"phrase": ph, "rects": rects, "in": round(max(a + 0.7, t_hl) + 0.7 * k, 3)})
                        k += 1
                s["highlights"] = hls
                if hls and v.get("size"):
                    r0 = hls[0]["rects"][0]
                    s["focus"] = [round((r0[0] + r0[2] / 2) / v["size"][0], 3), round((r0[1] + r0[3] / 2) / v["size"][1], 3)]
                if v.get("extra"):  # montage: the extra articles fly in one after another over the shot
                    cards = [{"src": v["src"], "highlights": hls, "focus": s.get("focus"), "in": a}]
                    n_c = 1 + len(v["extra"])
                    for k, ex in enumerate(v["extra"], 1):
                        t_in = round(a + (z - a) * k / n_c, 3)
                        hl2 = [{"phrase": ph, "rects": r, "in": round(t_in + 0.5, 3)} for ph, r in (ex.get("boxes") or {}).items() if r][:1]
                        foc = None
                        if hl2 and ex.get("size"):
                            r0 = hl2[0]["rects"][0]
                            foc = [round((r0[0] + r0[2] / 2) / ex["size"][0], 3), round((r0[1] + r0[3] / 2) / ex["size"][1], 3)]
                        cards.append({"src": ex["src"], "highlights": hl2, "focus": foc, "in": t_in, "url": ex["url"]})
                    s["cards"] = cards
            elif kind == "graphic":  # simple motion graphics: each item on its word
                gr = b.get("graphic") or {}
                s.update({"kind": "graphic", "graphic": {"kind": gr.get("kind", "connect"), "items": [
                    {"text": it["text"], "in": round(max(a + 0.15, min(z - 0.5, it.get("t", a) + 0.05)), 3)}
                    for it in gr.get("items", [])][:6]}})
            elif kind == "cta":
                s.update({"kind": "cta", "bg_src": v.get("bg_src"), "words": self.cta_words(b, a, z)})
            elif kind == "follow":
                s.update({"kind": "follow", "avatar_src": v.get("avatar_src"), "bg_src": v.get("bg_src"),
                          "bg_is_profile": v.get("bg_is_profile", False),
                          "handle": self.handle})
            elif kind == "facecam":
                s.update({"kind": "facecam", "src": None, "punches": self.facecam_punches(a, z, i), "keep_scales": True})
            s["kind"] = s.get("kind", kind)
            if trans:
                s["trans"] = trans
            if b.get("sfx") and b["sfx"] != "none":
                s["sfx"] = b["sfx"]
            if s["kind"] in ("image", "news", "map", "facecam", "graphic"):
                tl = self.text_layer(b, z)
                if tl and s["kind"] in ("map", "graphic", "news") and tl["preset"] not in ("headline_top", "caption_box"):
                    # top band, clear of the places / the cards / the article's text (an impact line over a news
                    # card's paragraphs was unreadable, ER 34 s)
                    tl["y"] = {"map": 0.2, "graphic": 0.17, "news": 0.11}[s["kind"]]
                    if s["kind"] == "news":
                        tl["scale"] = 0.8
                if tl:
                    s.setdefault("layers", []).append(tl)
            s.setdefault("layers", [])
            if b.get("stamp") and s["kind"] in ("image", "news", "map"):
                s["layers"].append({"type": "stamp", "text": b["stamp"]["text"], "out": round(z - 0.1, 3),
                                    "in": round(max(a + 0.3, min(z - 1.0, b["stamp"].get("t", a + 1))), 3)})
            shots.append(s)
        credits = []
        if any(s["kind"] == "map" for s in shots):
            credits = sorted({c for c in [self.P.L["basemap"].get("credit")] if c})
        return {"version": 1, "mode": "story", "duration": round(self.dur, 3), "offset": 0, "fps": self.P.fps,
                "width": self.W, "height": self.H, "look": "story_map", "handle": self.handle,
                "audio": next((p.name for p in self.ep.glob("narration.*")), None),
                "geometry": self.P.out_geom, "shots": shots, "credits": credits}


def brand_handle(episode=None):
    """The creator's handle: inputs/<episode>/brand.json first (a reel made for another creator), then inputs/brand/."""
    for f in ([ROOT / "inputs" / Path(episode).name / "brand.json"] if episode else []) + [ROOT / "inputs" / "brand" / "brand.json"]:
        if f.exists() and read_json(f).get("handle"):
            return read_json(f)["handle"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--handle", help="creator handle for the watermark / end card (default inputs/brand/brand.json)")
    a = ap.parse_args()
    tokens = read_json(ROOT / "style/tokens.json")
    sc = StoryPlanner(a.episode, tokens, a.handle or brand_handle(a.episode)).plan()
    validate(sc)
    out = write_json(Path(a.episode) / "scenes.json", sc)
    kinds = {}
    for s in sc["shots"]:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    print(f"[story_plan] {len(sc['shots'])} shots {kinds} -> {out}")


if __name__ == "__main__":
    main()
