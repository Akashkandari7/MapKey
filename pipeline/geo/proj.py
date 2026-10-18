"""Web Mercator (EPSG:3857) helpers. The renderer works in Web Mercator metres: x right, y up."""
import math

R = 6378137.0


def to_3857(lon, lat):
    lat = max(-85.0, min(85.0, lat))
    return math.radians(lon) * R, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R


def from_3857(x, y):
    return math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)


def geom_to_3857(geom):
    from shapely.ops import transform
    return transform(lambda x, y, z=None: _vec(x, y), geom)


def _vec(xs, ys):
    import numpy as np
    xs, ys = np.asarray(xs, float), np.clip(np.asarray(ys, float), -85, 85)
    return np.radians(xs) * R, np.log(np.tan(np.pi / 4 + np.radians(ys) / 2)) * R
