"""Review loop (runs after render, before mix): check the planned + rendered reel, fix what can be fixed, report.

    python -m pipeline.review --episode work/<ep> [--no-fix]

Checks (every text box is computed on screen every 0.25 s with the renderer's camera maths):
  coverage     every place/route spoken during the map shows on screen (its own layer, a pin at the same spot, or the
               reel's main place); places said on camera are listed, not required
  geo          geocoding items that need a human look (approximate points, lines built from station lists)
  safe_area    no text or label outside the Instagram safe area (tokens format.safe_area)
  covers       no title/stat sits on a drawn line or a label (the highlighted thing must stay visible)
  collisions   labels and stat blocks never overlap each other
  sync         every graphic lands on its word: |in - word time - style offset| <= 0.3 s
  pacing       a new visual event at least every few seconds while the map is on (reference: every 1-2 s)
  camera       move lengths / zoom factors inside the look's measured range
  colour       map frames' Lab brightness inside the subject and at the frame bottom within the reference range
Fixes (scenes.json, never on layers with "locked": true): add missing pins, move text inside the safe area or off a
line/label (mirror / shift), re-pick a label side, snap graphics to their word. The caller re-renders and re-checks.
Writes work/<ep>/review.json and review.md.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from pipeline.common import ROOT, read_json, write_json
from pipeline.plan import camera_at

STEP = 0.25


class Reviewer:
    def __init__(self, ep):
        self.ep = Path(ep)
        self.T = read_json(ROOT / "style/tokens.json")
        self.sc = read_json(self.ep / "scenes.json")
        self.ents = {e["id"]: e for e in read_json(self.ep / "entities.json")["entities"]}
        self.ents_doc = read_json(self.ep / "entities.json")
        self.geo = read_json(self.ep / "geo.json")
        self.W, self.H = self.sc["width"], self.sc["height"]
        self.L = self.T["looks"][self.sc["look"]]
        sa = self.T["format"]["safe_area"]
        self.safe = (sa["left"], sa["top"], self.W - sa["right"], self.H - sa["bottom"])
        self.issues, self.fixed = [], []

    # ---------- geometry of what is on screen ----------
    def cam(self, shot, t):
        return camera_at(shot["camera"], t, shot.get("drift_per_s", 0))

    def to_px(self, c, X, Y):
        return (X - c[0]) / c[2] + self.W / 2, (c[1] - Y) / c[2] + self.H / 2

    def visible_times(self, shot, layer):
        t0 = layer.get("in", layer["lines"][0]["in"] if layer.get("lines") else shot["start"])
        t1 = layer.get("out") or shot["end"]
        return [t for t in np.arange(t0, min(t1, shot["end"]), STEP)]

    def text_box(self, shot, layer, t):
        """Screen box (x0, y0, x1, y1) of a title / stat / pin label at time t (same layout rules as mapani.js)."""
        c = self.cam(shot, t) if shot["kind"] == "map" else (0, 0, 1)
        if layer["type"] == "title":
            st = self.L[layer.get("style", "title_on_map")] if layer.get("style") in self.L else self.L["title_on_map"]
            k = layer["mpp_in"] / c[2] if "X" in layer else 1
            x, y = self.to_px(c, layer["X"], layer["Y"]) if "X" in layer else (layer["x"], layer["y"])
            w, h = len(layer["text"]) * 0.56 * st["size"] * k, st["size"] * k
            return (x - w / 2, y - h, x + w / 2, y + 0.2 * h)
        if layer["type"] == "stat":
            S = self.L["stat"]
            world = "X" in layer and shot["kind"] == "map"
            x, y = self.to_px(c, layer["X"], layer["Y"]) if world else (layer["x"], layer["y"])
            scale = 1 if world or shot["kind"] == "map" else self.T["facecam"]["stat_box"]["size_scale"]
            hs = [(ln.get("h_world") / c[2]) if world and ln.get("h_world") else S[ln["style"]]["size"] * scale
                  for ln in layer["lines"]]
            gap = self.T["plan"]["stat_line_gap"] * self.H * ((layer.get("mpp_in", c[2]) / c[2]) if world else 1)
            w = max(len(ln["text"]) * 0.58 * h for ln, h in zip(layer["lines"], hs))
            tot = sum(hs) + gap * (len(hs) - 1)
            return (x - w / 2, y - tot / 2, x + w / 2, y + tot / 2)
        if layer["type"] == "pin":
            pill, st = self.L["label_pill"], self.L["station"]
            X, Y = self.sc["geometry"][layer["geom"]]["coords"]
            x, y = self.to_px(c, X, Y)
            w = len(layer["label"]) * 0.56 * pill["size"] + 2 * pill["pad_x"]
            h, gap = pill["size"] + 2 * pill["pad_y"], st["dot_radius"] + 10
            return {"above": (x - w / 2, y - gap - h, x + w / 2, y - gap), "below": (x - w / 2, y + gap, x + w / 2, y + gap + h),
                    "right": (x + gap, y - h / 2, x + gap + w, y + h / 2),
                    "left": (x - gap - w, y - h / 2, x - gap, y + h / 2)}[layer.get("side", "above")]
        return None

    def route_px(self, shot, layer, t):
        """Screen points of the part of a route drawn by time t (dense enough to test against boxes)."""
        g = self.sc["geometry"][layer["geom"]]["coords"]
        u = min(1.0, max(0.0, (t - layer["in"]) / layer.get("draw_s", 1)))
        if u <= 0:
            return np.zeros((0, 2))
        c = self.cam(shot, t)
        pts = np.array([self.to_px(c, *p) for p in g])
        if len(pts) > 1:  # densify to ~6 px so a thin line crossing a box is caught
            seg = np.hypot(*np.diff(pts, axis=0).T)
            dense = [pts[0]]
            for p, q, d in zip(pts, pts[1:], seg):
                n = max(1, int(d / 6))
                dense += [p + (q - p) * k / n for k in range(1, n + 1)]
            pts = np.array(dense)
        return pts[: max(1, int(len(pts) * u))]

    @staticmethod
    def overlap(a, b, shrink=0.0):
        if shrink:
            dx, dy = (a[2] - a[0]) * shrink, (a[3] - a[1]) * shrink
            a = (a[0] + dx, a[1] + dy, a[2] - dx, a[3] - dy)
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    def outside_by(self, box):
        """How far (dx, dy) the box must move to be inside the safe area (0, 0 if inside)."""
        x0, y0, x1, y1 = self.safe
        dx = (x0 - box[0]) if box[0] < x0 else ((x1 - box[2]) if box[2] > x1 else 0)
        dy = (y0 - box[1]) if box[1] < y0 else ((y1 - box[3]) if box[3] > y1 else 0)
        return dx, dy

    def issue(self, check, level, msg, **kw):
        self.issues.append({"check": check, "level": level, "msg": msg, **kw})

    # ---------- checks ----------
    def check_coverage(self):
        map_shots = [s for s in self.sc["shots"] if s["kind"] == "map"]
        cam_shots = [s for s in self.sc["shots"] if s["kind"] == "facecam"]
        shown = {l.get("ent") for s in map_shots for l in s.get("layers", [])}
        geo = {it["id"]: it for it in self.geo["items"]}
        main = self.ents_doc.get("main_place")
        for e in self.ents.values():
            if e["type"] not in ("place", "route"):
                continue
            in_map = any(s["start"] <= e["start"] < s["end"] for s in map_shots)
            if not in_map:
                if any(s["start"] <= e["start"] < s["end"] for s in cam_shots):
                    self.issue("coverage", "info", f"{e.get('display') or e.get('name_en')} is said on camera (not on the map)")
                continue
            if e["id"] in shown or e["id"] == main or self.same_spot_as_pin(e, map_shots):
                continue
            g = geo.get(e["id"], {}).get("resolved")
            self.issue("coverage", "fail", f"'{e.get('display') or e.get('name_en')}' is spoken at {e['start']:.1f}s but "
                       f"not shown" + ("" if g else " (no geometry found)"), ent=e["id"], fixable=bool(g))

    def same_spot_as_pin(self, e, shots):
        from shapely.geometry import Point, shape
        from pipeline.geo.proj import geom_to_3857
        it = next((x for x in self.geo["items"] if x["id"] == e["id"] and x["resolved"]), None)
        if not it:
            return False
        p = geom_to_3857(shape(it["resolved"]["geometry"])).representative_point()
        return any(l["type"] == "pin" and Point(self.sc["geometry"][l["geom"]]["coords"]).distance(p) < 400
                   for s in shots for l in s.get("layers", []))

    def check_india(self):
        """India rule: no shape except India's own may cover the Survey of India outline (incl. Aksai Chin,
        Arunachal, Gilgit-Baltistan, PoK). Measured on the exact geometry the renderer draws."""
        from shapely.geometry import MultiPolygon, Polygon
        from pipeline.geo.india import is_india, outline_3857
        ind = outline_3857()
        for gid, g in self.sc["geometry"].items():
            if g["kind"] != "polygon":
                continue
            shp = MultiPolygon([Polygon(p[0], p[1:]) for p in g["coords"]]).buffer(0)
            if is_india(shp, lonlat=False):
                continue
            km2 = shp.intersection(ind).area / 1e6 * 0.75  # Web Mercator inflates area ~1.3x at India's latitudes
            if km2 > 0.5:
                self.issue("india", "fail", f"'{g['name']}' covers {km2:.1f} km2 of India (SoI outline)", gid=gid)

    def check_geo(self):
        for r in self.geo.get("needs_review", []):
            self.issue("geo", "warn", f"'{r['name']}' placed by {r['method']} (confidence {r['confidence']}): check it")
        for m in self.geo.get("missing", []):
            self.issue("geo", "warn", f"{m}: no geometry found, not drawn")

    def text_layers(self):
        for s in self.sc["shots"]:
            for i, l in enumerate(s.get("layers", [])):
                if l["type"] in ("title", "stat", "pin"):
                    yield s, i, l

    def check_safe_area(self):
        for s, i, l in self.text_layers():
            worst = (0, 0, None)
            for t in self.visible_times(s, l):
                dx, dy = self.outside_by(self.text_box(s, l, t))
                if abs(dx) + abs(dy) > abs(worst[0]) + abs(worst[1]):
                    worst = (dx, dy, t)
            if abs(worst[0]) + abs(worst[1]) > 8:
                self.issue("safe_area", "fail", f"{self.name(l)} goes {abs(worst[0]) + abs(worst[1]):.0f}px outside the "
                           f"safe area at {worst[2]:.1f}s", shot=s["id"], layer=i, shift=(worst[0], worst[1]), t=worst[2])

    def check_covers_and_collisions(self):
        for s in self.sc["shots"]:
            if s["kind"] != "map":
                continue
            layers = s.get("layers", [])
            for i, l in enumerate(layers):
                if l["type"] not in ("title", "stat"):
                    continue
                for t in self.visible_times(s, l)[::4]:
                    box = self.text_box(s, l, t)
                    hit = next((r for r in layers if r["type"] == "route" and r["in"] <= t and
                                any(box[0] < x < box[2] and box[1] < y < box[3] for x, y in self.route_px(s, r, t))), None)
                    if hit:
                        self.issue("covers", "fail", f"{self.name(l)} covers the {self.sc['geometry'][hit['geom']]['name']} "
                                   f"line at {t:.1f}s", shot=s["id"], layer=i, t=t)
                        break
            texts = [(i, l) for i, l in enumerate(layers) if l["type"] in ("pin", "stat", "title")]
            for a in range(len(texts)):
                for b in range(a + 1, len(texts)):
                    (i, la), (j, lb) = texts[a], texts[b]
                    common = sorted(set(round(t, 2) for t in self.visible_times(s, la)) &
                                    set(round(t, 2) for t in self.visible_times(s, lb)))
                    for t in common[::2]:
                        if self.overlap(self.text_box(s, la, t), self.text_box(s, lb, t), shrink=0.05):
                            self.issue("collisions", "fail", f"{self.name(la)} overlaps {self.name(lb)} at {t:.1f}s",
                                       shot=s["id"], layer=j if lb["type"] == "pin" else i, other=i, t=t)
                            break

    def check_sync(self):
        M = self.T["motion"]
        offs = {"route": 0.0, "pin": M["label_in"]["start_after_word"], "stat": M["number_in"]["start_after_word"],
                "title": M["text_in"]["start_after_word"]}
        for s in self.sc["shots"]:
            for i, l in enumerate(s.get("layers", [])):
                items = [(l, l.get("ent"))] if l["type"] in ("route", "pin", "title") else \
                    [(ln, ln.get("ent")) for ln in l.get("lines", [])]
                for obj, ent in items:
                    e = self.ents.get(ent)
                    if not e or (l["type"] == "title" and obj["in"] <= s["start"] + 0.5) or \
                            (l["type"] == "route" and obj["in"] <= s["start"] + 0.5):
                        continue  # hook title / hook line start with the reel on purpose
                    d = obj["in"] - (e["start"] + offs[l["type"]])
                    if abs(d) > 0.3:
                        self.issue("sync", "fail", f"{self.name(l)} lands {d:+.2f}s off its word '{e['quote']}'",
                                   shot=s["id"], layer=i, ent=ent, want=round(e["start"] + offs[l["type"]], 3))

    def check_pacing(self):
        for s in self.sc["shots"]:
            if s["kind"] != "map":
                continue
            ev = sorted({round(l["in"], 2) for l in s.get("layers", []) if "in" in l} |
                        {round(ln["in"], 2) for l in s.get("layers", []) for ln in l.get("lines", [])} |
                        {round(k.get("t0", k["t"]), 2) for k in s["camera"][1:]} | {s["start"], s["end"]})
            gaps = np.diff(ev)
            if len(gaps) and gaps.max() > 4.5:
                k = int(np.argmax(gaps))
                self.issue("pacing", "warn", f"nothing new on screen for {gaps.max():.1f}s ({ev[k]:.1f}-{ev[k + 1]:.1f}s); "
                           f"reference: a new graphic every 1-2 s")

    def check_camera(self):
        cam = self.T["motion"]["camera"][self.sc["look"]]
        for s in self.sc["shots"]:
            if s["kind"] != "map":
                continue
            prev = s["camera"][0]
            for k in s["camera"][1:]:
                dur = k["t"] - k.get("t0", prev["t"])
                z = prev["mpp"] / k["mpp"]
                if not (cam["move_s"][0] - 0.5 <= dur <= cam["move_s"][1] + 0.5):
                    self.issue("camera", "warn", f"move at {k.get('t0', 0):.1f}s lasts {dur:.1f}s (look: {cam['move_s']} s)")
                if not (1 / 3.5 <= z <= 3.5):
                    self.issue("camera", "warn", f"move at {k.get('t0', 0):.1f}s zooms x{z:.2f} (too abrupt)")
                prev = k

    def check_colour(self):
        import cv2
        raw = self.ep / "video_raw.mp4"
        R = self.T["review"]
        if not raw.exists():
            return
        cap = cv2.VideoCapture(str(raw))
        ins, bot = [], []
        for s in self.sc["shots"]:
            if s["kind"] != "map":
                continue
            for t in np.arange(s["start"] + 1, s["end"] - 0.5, 2.0):
                cap.set(cv2.CAP_PROP_POS_MSEC, (t - self.sc.get("offset", 0)) * 1000)
                ok, f = cap.read()
                if not ok:
                    continue
                lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
                h, w = lab.shape[:2]
                ins.append(lab[int(.3 * h):int(.65 * h), int(.25 * w):int(.75 * w)].reshape(-1, 3).mean(0))
                bot.append(lab[int(.82 * h):, :].reshape(-1, 3).mean(0))
        cap.release()
        if not ins:
            return
        mi, mb = np.median(ins, axis=0), np.median(bot, axis=0)
        if not R["inside_L"][0] <= mi[0] <= R["inside_L"][1]:
            self.issue("colour", "warn", f"map inside brightness L {mi[0]:.0f} (reference {R['inside_L']})")
        if not R["bottom_L"][0] <= mb[0] <= R["bottom_L"][1]:
            self.issue("colour", "warn", f"frame bottom brightness L {mb[0]:.0f} (reference {R['bottom_L']})")
        self.colour = {"inside_Lab": [round(float(v), 1) for v in mi], "bottom_Lab": [round(float(v), 1) for v in mb]}

    def name(self, l):
        if l["type"] == "pin":
            return f"label '{l['label']}'"
        if l["type"] == "stat":
            return "stat '" + " / ".join(x["text"] for x in l["lines"]) + "'"
        if l["type"] == "title":
            return f"title '{l['text']}'"
        return f"{l['type']} {l.get('geom', '')}"

    def run_checks(self):
        self.issues = []
        self.check_india(); self.check_coverage(); self.check_geo(); self.check_safe_area(); self.check_covers_and_collisions()
        self.check_sync(); self.check_pacing(); self.check_camera(); self.check_colour()
        return self.issues

    # ---------- fixes ----------
    def move_layer(self, s, l, dx, dy, t):
        """Shift a text layer by (dx, dy) screen px as seen at time t (world-anchored text moves on the map)."""
        if "X" in l and s["kind"] == "map":
            m = self.cam(s, t)[2]
            l["X"] = round(l["X"] + dx * m, 1)
            l["Y"] = round(l["Y"] - dy * m, 1)
        l["x"], l["y"] = round(l.get("x", 0) + dx, 1), round(l.get("y", 0) + dy, 1)

    def layer_ok(self, s, l):
        for t in self.visible_times(s, l)[::2]:
            b = self.text_box(s, l, t)
            dx, dy = self.outside_by(b)
            if abs(dx) + abs(dy) > 8:
                return False
            for r in s.get("layers", []):
                if r is l:
                    continue
                if r["type"] == "route" and r["in"] <= t and l["type"] != "pin" and \
                        any(b[0] < x < b[2] and b[1] < y < b[3] for x, y in self.route_px(s, r, t)):
                    return False
                if r["type"] in ("pin", "stat", "title") and t in [round(x, 2) for x in self.visible_times(s, r)] and \
                        self.overlap(b, self.text_box(s, r, t), shrink=0.05):
                    return False
        return True

    def apply_fixes(self):
        shots = {s["id"]: s for s in self.sc["shots"]}
        done = set()
        for it in self.issues:
            if it["level"] != "fail":
                continue
            key = (it["check"], it.get("shot"), it.get("layer"), it.get("ent"))
            if key in done:
                continue
            done.add(key)
            if it["check"] == "india":
                self.clip_india(it["gid"])
                self.fixed.append(f"clipped to India's border: {it['msg']}")
                continue
            if it["check"] == "coverage" and it.get("fixable"):
                self.add_pin(it["ent"])
                continue
            s = shots.get(it.get("shot"))
            if s is None or it.get("layer") is None:
                continue
            l = s["layers"][it["layer"]]
            if l.get("locked"):
                self.fixed.append(f"left alone (locked): {it['msg']}")
                continue
            if it["check"] == "sync":
                tgt = l if l["type"] in ("route", "pin", "title") else next(x for x in l["lines"] if x.get("ent") == it["ent"])
                tgt["in"] = it["want"]
                self.fixed.append(f"snapped to its word: {it['msg']}")
            elif it["check"] == "safe_area" and l["type"] == "pin":
                # a label's dot marks the real place and can't move: put the pill on another side instead
                if self.relocate(s, l, it["t"]):
                    self.fixed.append(f"label turned to the '{l['side']}' side: {it['msg']}")
                else:
                    it["level"] = "warn"
                    it["msg"] += " (no side fits - check by eye)"
            elif it["check"] == "safe_area":
                self.move_layer(s, l, it["shift"][0] * 1.1, it["shift"][1] * 1.1, it["t"])
                self.fixed.append(f"moved inside the safe area: {it['msg']}")
            elif it["check"] in ("covers", "collisions"):
                if self.relocate(s, l, it["t"]):
                    self.fixed.append(f"moved clear: {it['msg']}")
                else:
                    it["level"] = "warn"
                    it["msg"] += " (no free spot found - check by eye)"
        return self.fixed

    def relocate(self, s, l, t):
        if l["type"] == "pin":
            for side in ("above", "below", "right", "left"):
                if side == l.get("side"):
                    continue
                trial = dict(l, side=side)
                if self.layer_ok(s, trial):
                    l["side"] = side
                    return True
            return False
        b = self.text_box(s, l, t)
        w, h = b[2] - b[0], b[3] - b[1]
        for dx, dy in ((-w * 0.8, 0), (w * 0.8, 0), (0, -h * 1.2), (0, h * 1.2), (-w * 0.8, -h), (w * 0.8, h)):
            trial = copy.deepcopy(l)
            self.move_layer(s, trial, dx, dy, t)
            if self.layer_ok(s, trial):
                l.update(trial)
                return True
        return False

    def clip_india(self, gid):
        from shapely.geometry import MultiPolygon, Polygon
        from pipeline.geo.india import clip_out
        g = self.sc["geometry"][gid]
        shp = clip_out(MultiPolygon([Polygon(p[0], p[1:]) for p in g["coords"]]).buffer(0), lonlat=False)
        polys = list(getattr(shp, "geoms", [shp]))
        g["coords"] = [[[[round(x, 1), round(y, 1)] for x, y in ring.coords] for ring in [p.exterior, *p.interiors]]
                       for p in polys if p.geom_type == "Polygon"]

    def add_pin(self, ent_id):
        from shapely.geometry import shape
        from pipeline.geo.proj import geom_to_3857
        e = self.ents[ent_id]
        it = next(x for x in self.geo["items"] if x["id"] == ent_id)
        g = geom_to_3857(shape(it["resolved"]["geometry"]))
        if g.geom_type in ("LineString", "MultiLineString"):
            return
        p = g.representative_point()
        s = next(s for s in self.sc["shots"] if s["kind"] == "map" and s["start"] <= e["start"] < s["end"])
        gid = "g_" + ent_id
        self.sc["geometry"][gid] = {"kind": "point", "name": e.get("name_en"), "coords": [round(p.x, 1), round(p.y, 1)]}
        pin = {"type": "pin", "ent": ent_id, "geom": gid, "label": e.get("display") or e.get("name_en"),
               "in": round(e["start"] + self.T["motion"]["label_in"]["start_after_word"], 3), "side": "above"}
        s["layers"].append(pin)
        self.relocate(s, pin, pin["in"] + 1) if not self.layer_ok(s, pin) else None
        self.fixed.append(f"added a missing label for '{pin['label']}' at {pin['in']:.1f}s")

    def save(self, fixed_any):
        if fixed_any:
            bak = self.ep / "scenes.before_review.json"
            if not bak.exists():
                write_json(bak, read_json(self.ep / "scenes.json"))
            write_json(self.ep / "scenes.json", self.sc)

    def report(self, rounds):
        fails = [i for i in self.issues if i["level"] == "fail"]
        warns = [i for i in self.issues if i["level"] == "warn"]
        rep = {"rounds": rounds, "fixed": self.fixed, "fails": fails, "warnings": warns,
               "info": [i for i in self.issues if i["level"] == "info"], "colour": getattr(self, "colour", None)}
        write_json(self.ep / "review.json", rep)
        lines = [f"# Review: {self.ep.name}", "",
                 f"**{'PASS' if not fails else 'NEEDS A LOOK'}**: {len(fails)} problems left, {len(warns)} warnings, "
                 f"{len(self.fixed)} fixed automatically in {rounds} round(s).", ""]
        for title, items in (("Fixed automatically", self.fixed), ("Problems left", [i["msg"] for i in fails]),
                             ("Warnings", [i["msg"] for i in warns]), ("Info", [i["msg"] for i in rep["info"]])):
            if items:
                lines += [f"## {title}", ""] + [f"- {x}" for x in items] + [""]
        (self.ep / "review.md").write_text("\n".join(lines), encoding="utf-8")
        return rep


def review(ep, fix=True, rerender=None, max_rounds=2):
    """Check -> fix scenes.json -> re-render (callback) -> check again, up to max_rounds; report the final state."""
    all_fixed, rounds = [], 0
    while rounds < max_rounds:
        rv = Reviewer(ep)
        rv.run_checks()
        rounds += 1
        fixed = rv.apply_fixes() if fix else []
        rv.save(bool(fixed))
        all_fixed += fixed
        if not fixed:
            break
        print(f"[review] round {rounds}: fixed {len(fixed)} issue(s)" + (", re-rendering" if rerender else ""))
        if rerender:
            rerender()
    final = Reviewer(ep)
    final.run_checks()
    final.fixed = list(dict.fromkeys(all_fixed))  # a locked layer is reported once, not once per round
    return final.report(rounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--no-fix", action="store_true")
    a = ap.parse_args()
    rep = review(a.episode, fix=not a.no_fix)
    print(Path(a.episode, "review.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
