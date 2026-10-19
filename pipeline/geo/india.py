"""The Survey of India outline, for the India rule: no other country's or province's shape may cover any of it.

Simplifying a long border (China's is ~3,500 km) moves it by a few hundred metres either way, so every non-India
polygon is clipped by this outline AFTER it is simplified - in geocode (lon/lat) and in the planner (Web Mercator).
"""
from functools import lru_cache

from pipeline.common import ROOT


@lru_cache(maxsize=1)
def outline_lonlat():
    import geopandas as gpd
    c = gpd.read_file(ROOT / "data/geo/layers.gpkg", layer="countries")
    return c[c.ADM0_A3 == "IND"].geometry.iloc[0]


@lru_cache(maxsize=1)
def outline_3857():
    from pipeline.geo.proj import geom_to_3857
    return geom_to_3857(outline_lonlat())


def is_india(geom, lonlat=True):
    """True if the shape IS India (its own outline), not just touching it."""
    ref = outline_lonlat() if lonlat else outline_3857()
    return geom.geom_type in ("Polygon", "MultiPolygon") and geom.intersection(ref).area > 0.9 * ref.area


def clip_out(geom, lonlat=True):
    """Remove SoI India from a polygon (no-op for points/lines and for India itself)."""
    if geom.geom_type not in ("Polygon", "MultiPolygon") or is_india(geom, lonlat):
        return geom
    ref = outline_lonlat() if lonlat else outline_3857()
    if not geom.intersects(ref):
        return geom
    return geom.difference(ref)
