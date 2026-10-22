"""Satellite basemap from free, commercial-use-OK open data: Copernicus Sentinel-2 L2A true colour (10 m).

    python -m pipeline.imagery --bbox 76.85,28.30,77.25,28.65 --res 20 --out data/geo/cache/imagery/test

What it does: finds cloud-free Sentinel-2 scenes (Element84 Earth Search STAC catalogue of the public AWS archive,
no login), reads only the needed area at the needed resolution straight from the cloud-optimised GeoTIFFs, reprojects
to Web Mercator (EPSG:3857, the projection the renderer draws in), and takes the per-pixel MEDIAN of the clearest
scenes. The median removes leftover clouds, cloud shadows and the Oct-Dec north-Indian smog that single scenes carry.

Output (cached by bbox + resolution): <out>.jpg (RGB, quality 92) + <out>.json
    {"bounds_3857": [minx, miny, maxx, maxy], "res_m": 20, "size": [w, h], "scenes": [...ids], "credit": "..."}

Licence: Copernicus Sentinel data are free for any use incl. commercial; credit "Contains modified Copernicus
Sentinel data <year>". Logged in data/geo/SOURCES.md.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from pipeline.common import ROOT, sha

STAC = "https://earth-search.aws.element84.com/v1"
CACHE = ROOT / "data/geo/cache/imagery"
# clearest season over South Asia (after monsoon haze, before spring dust); smog months are diluted by the median
DEFAULT_DATES = "2024-10-15/2025-04-15"
R = 6378137.0


def lonlat_to_3857(lon, lat):
    x = math.radians(lon) * R
    y = math.log(math.tan(math.pi / 4 + math.radians(max(-85, min(85, lat))) / 2)) * R
    return x, y


def search(bbox, dates, max_cloud=5, limit=60):
    from pystac_client import Client
    c = Client.open(STAC)
    items = list(c.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=dates,
                          query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=limit).items())
    return sorted(items, key=lambda i: (i.properties["eo:cloud_cover"], i.properties.get("s2:nodata_pixel_percentage", 0)))


def build(bbox, res_m, out, dates=DEFAULT_DATES, per_tile=4, force=False):
    """bbox = [min_lon, min_lat, max_lon, max_lat]; res_m = metres per pixel in Web Mercator units."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    out = Path(out)
    meta_f = out.with_suffix(".json")
    if meta_f.exists() and out.with_suffix(".jpg").exists() and not force:
        return json.loads(meta_f.read_text())
    x0, y0 = lonlat_to_3857(bbox[0], bbox[1])
    x1, y1 = lonlat_to_3857(bbox[2], bbox[3])
    w, h = int(math.ceil((x1 - x0) / res_m)), int(math.ceil((y1 - y0) / res_m))
    if w * h > 60e6:
        raise SystemExit(f"[imagery] {w}x{h} px is too large; use a coarser --res")
    mosaic = res_m >= 80
    if mosaic:
        # coarse views: no median (dozens of scenes over a whole region would need gigabytes of RAM); a mosaic of the
        # clearest scenes instead, 3 per tile: one scene covers only part of its tile (orbit swath edge), and the
        # holes it leaves showed the blurry level below as hard-edged patches
        per_tile = 3
    items = search(bbox, dates)
    # keep the clearest few scenes of every MGRS tile that touches the area
    by_tile = {}
    for it in items:
        tile = it.properties.get("s2:mgrs_tile") or it.id.split("_")[1]
        by_tile.setdefault(tile, [])
        if len(by_tile[tile]) < per_tile:
            by_tile[tile].append(it)
    chosen = [it for v in by_tile.values() for it in v]
    if not chosen:
        raise SystemExit("[imagery] no clear Sentinel-2 scenes found; widen --dates")
    print(f"[imagery] {w}x{h} px at {res_m} m, {len(chosen)} scenes from tiles {sorted(by_tile)}")
    stack = []
    grid = from_origin(x0, y0 + h * res_m, res_m, res_m)  # our output grid; the VRT warps each scene onto it
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                       GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE")
    def read(it):
        with env:
            try:
                # read the COG's own overview nearest the target resolution (20/40/80/160 m), not the 10 m data:
                # a 160 m mosaic then moves ~250x fewer bytes (it took 35+ min at full resolution)
                ovr = int(max(-1, min(3, math.floor(math.log2(res_m / 10)) - 1)))
                with rasterio.open(it.assets["visual"].href, **({"OVERVIEW_LEVEL": ovr} if ovr >= 0 else {})) as src, WarpedVRT(
                        src, crs="EPSG:3857", transform=grid, width=w, height=h, src_nodata=0, nodata=0,
                        resampling=Resampling.average) as vrt:
                    arr = vrt.read(indexes=[1, 2, 3])
                print(f"   {it.id} cloud {it.properties['eo:cloud_cover']:.2f}")
                return arr
            except Exception as e:  # noqa: BLE001  (one bad scene must not sink the mosaic)
                print(f"   skip {it.id}: {e}", file=sys.stderr)
                return None

    # 3 scenes at a time: network-bound, and keeps the laptop's CPU mostly free
    from concurrent.futures import ThreadPoolExecutor
    if mosaic:  # clearest scene first, later scenes only fill what is still empty (low memory)
        med = np.zeros((3, h, w), np.uint8)
        with ThreadPoolExecutor(3) as ex:
            for arr in ex.map(read, chosen):
                if arr is not None:
                    have, new = med.sum(axis=0) > 0, arr.sum(axis=0) > 0
                    both = have & new
                    # scenes from different days differ in haze and sun: match each new scene's colour to the mosaic
                    # where they overlap (per-channel mean/std), so the patches it fills don't show as blocks
                    if both.sum() > 2000:
                        a = arr.astype(np.float32)
                        for ch in range(3):
                            ma, sa = a[ch][both].mean(), a[ch][both].std() + 1e-3
                            mm, sm = med[ch][both].mean(), med[ch][both].std() + 1e-3
                            a[ch] = np.where(new, (a[ch] - ma) / sa * sm + mm, 0)
                        arr = np.clip(a, 1, 255).astype(np.uint8) * new
                    empty = ~have & new
                    med[:, empty] = arr[:, empty]
        holes = med.sum(axis=0) == 0
    else:
        with ThreadPoolExecutor(3) as ex:
            stack = [a for a in ex.map(read, chosen) if a is not None]
        data = np.stack(stack).astype(np.float32)          # scenes x 3 x h x w
        valid = data.sum(axis=1, keepdims=True) > 0         # 0,0,0 = outside the scene footprint
        data = np.where(valid, data, np.nan)
        med = np.nanmedian(data, axis=0)                    # per-pixel median over the scenes that cover it
        holes = np.isnan(med).any(axis=0)
        med = np.nan_to_num(med, nan=0).clip(0, 255).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(np.moveaxis(med, 0, -1)).save(out.with_suffix(".jpg"), quality=92)
    years = sorted({it.properties["datetime"][:4] for it in chosen})
    meta = {"bounds_3857": [x0, y0, x1, y1], "bbox_lonlat": list(bbox), "res_m": res_m, "size": [w, h],
            "scenes": [it.id for it in chosen], "holes_pct": round(float(holes.mean()) * 100, 2),
            "credit": f"Contains modified Copernicus Sentinel data {'-'.join(years)}"}
    meta_f.write_text(json.dumps(meta, indent=1))
    print(f"[imagery] wrote {out.with_suffix('.jpg')} ({meta['holes_pct']}% empty)")
    return meta


def for_view(bbox, res_m, need=None):
    """Cached path for a bbox/resolution. An existing mosaic at the same resolution that already covers `need` (the area
    the camera really sees; default the bbox) is reused: a re-planned camera usually moves a little, and that must not
    trigger another download. New downloads use the full (padded) bbox."""
    need = need or bbox
    for meta_f in sorted(CACHE.glob("s2_*.json")):
        try:
            m = json.loads(meta_f.read_text())
        except ValueError:
            continue
        b = m.get("bbox_lonlat")
        if m.get("res_m") == res_m and b and b[0] <= need[0] and b[1] <= need[1] and b[2] >= need[2] and b[3] >= need[3] \
                and meta_f.with_suffix(".jpg").exists():
            return meta_f.with_suffix("")
    b = [round(v, 3) for v in bbox]
    return CACHE / f"s2_{sha(b, res_m)}"


# ---------- country / continent scale: NASA Blue Marble (public domain) ----------
# Sentinel-2 can't sensibly mosaic a whole country; for views coarser than ~300 m/px the basemap is NASA's Blue Marble
# (shaded relief + bathymetry), served as Web Mercator tiles by NASA GIBS (no login). NASA imagery is public domain.
GIBS = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/"
        "GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg")
WORLD = 20037508.342789244
BM_CREDIT = "NASA Blue Marble (GIBS)"


def bm_zoom(res_m):
    """GIBS zoom whose pixel size is closest to res_m (z4 = 9784 ... z8 = 611 m/px; z8 is the finest)."""
    return int(max(0, min(8, round(math.log2(2 * WORLD / 256 / res_m)))))


def bm_res(z):
    return 2 * WORLD / 256 / 2 ** z


def blue_marble(bbox, res_m, out, force=False):
    """Mosaic of Blue Marble tiles covering bbox (lon/lat) at the zoom matching res_m, cropped to the bbox."""
    import io
    from concurrent.futures import ThreadPoolExecutor
    import requests
    from PIL import Image
    out = Path(out)
    meta_f = out.with_suffix(".json")
    if meta_f.exists() and out.with_suffix(".jpg").exists() and not force:
        return json.loads(meta_f.read_text())
    z = bm_zoom(res_m)
    r = bm_res(z)
    (x0, y0), (x1, y1) = lonlat_to_3857(bbox[0], bbox[1]), lonlat_to_3857(bbox[2], bbox[3])
    px0, px1 = int(math.floor((x0 + WORLD) / r)), int(math.ceil((x1 + WORLD) / r))
    py0, py1 = int(math.floor((WORLD - y1) / r)), int(math.ceil((WORLD - y0) / r))
    n = 2 ** z
    tiles = [(tx, ty) for ty in range(py0 // 256, (py1 - 1) // 256 + 1) for tx in range(px0 // 256, (px1 - 1) // 256 + 1)]
    tdir = CACHE / "gibs" / str(z)
    tdir.mkdir(parents=True, exist_ok=True)

    def get(t):
        tx, ty = t
        f = tdir / f"{ty}_{tx % n}.jpg"
        if not f.exists():
            for attempt in range(3):
                try:
                    resp = requests.get(GIBS.format(z=z, y=ty, x=tx % n), timeout=30)
                    resp.raise_for_status()
                    f.write_bytes(resp.content)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    import time
                    time.sleep(3 * (attempt + 1))  # a dropped connection (Wi-Fi blip) gets a moment to come back
        return t, Image.open(io.BytesIO(f.read_bytes())).convert("RGB")

    canvas = Image.new("RGB", (px1 - px0, py1 - py0))
    with ThreadPoolExecutor(3) as ex:  # laptop rule: a few connections at a time
        for (tx, ty), im in ex.map(get, tiles):
            canvas.paste(im, (tx * 256 - px0, ty * 256 - py0))
    canvas.save(out.with_suffix(".jpg"), quality=92)
    meta = {"bounds_3857": [px0 * r - WORLD, WORLD - py1 * r, px1 * r - WORLD, WORLD - py0 * r],
            "bbox_lonlat": list(bbox), "res_m": res_m, "zoom": z, "size": list(canvas.size), "credit": BM_CREDIT}
    meta_f.write_text(json.dumps(meta, indent=1))
    print(f"[imagery] Blue Marble z{z}: {len(tiles)} tiles -> {out.with_suffix('.jpg')}")
    return meta


def level(bbox, res_m, need=None):
    """One basemap level: Blue Marble for coarse levels (>= 600 m/px), Sentinel-2 otherwise. Returns (path, meta)."""
    if res_m >= 600:
        need = need or bbox
        for meta_f in sorted(CACHE.glob("bm_*.json")):
            m = json.loads(meta_f.read_text())
            b = m["bbox_lonlat"]
            if m["res_m"] == res_m and b[0] <= need[0] and b[1] <= need[1] and b[2] >= need[2] and b[3] >= need[3]:
                return meta_f.with_suffix(""), m
        path = CACHE / f"bm_{sha([round(v, 3) for v in bbox], res_m)}"
        return path, blue_marble(bbox, res_m, path)
    path = for_view(bbox, res_m, need=need)
    return path, build(bbox, res_m, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat")
    ap.add_argument("--res", type=float, default=20)
    ap.add_argument("--out")
    ap.add_argument("--dates", default=DEFAULT_DATES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    bbox = [float(v) for v in a.bbox.split(",")]
    build(bbox, a.res, a.out or for_view(bbox, a.res), a.dates, force=a.force)


if __name__ == "__main__":
    main()
