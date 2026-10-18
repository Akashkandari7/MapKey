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
                with rasterio.open(it.assets["visual"].href) as src, WarpedVRT(
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
