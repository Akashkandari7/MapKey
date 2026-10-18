"""Download the open geo datasets into data/geo/raw/ (one time, ~60 MB), then build the database.

    python -m pipeline.geo.fetch_raw          # downloads what's missing, then runs build_layers

Sources and licences: data/geo/SOURCES.md.
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

from pipeline.common import ROOT

RAW = ROOT / "data" / "geo" / "raw"
DM = "https://raw.githubusercontent.com/datameet/maps/master/"
NE = "https://naciscdn.org/naturalearth/10m/"
FILES = {
    "india-composite.geojson": DM + "Country/india-composite.geojson",
    **{f"dm_states_Admin2.{ext}": DM + f"States/Admin2.{ext}" for ext in ("shp", "dbf", "shx", "prj", "cpg")},
    "ne_10m_admin_0_countries_ind.zip": NE + "cultural/ne_10m_admin_0_countries_ind.zip",
    "ne_10m_admin_1_states_provinces.zip": NE + "cultural/ne_10m_admin_1_states_provinces.zip",
    "ne_10m_populated_places_simple.zip": NE + "cultural/ne_10m_populated_places_simple.zip",
    "ne_10m_rivers_lake_centerlines.zip": NE + "physical/ne_10m_rivers_lake_centerlines.zip",
    "ne_10m_lakes.zip": NE + "physical/ne_10m_lakes.zip",
    "ne_10m_geography_regions_polys.zip": NE + "physical/ne_10m_geography_regions_polys.zip",
    "ne_10m_geography_marine_polys.zip": NE + "physical/ne_10m_geography_marine_polys.zip",
    "geonames_cities5000.zip": "https://download.geonames.org/export/dump/cities5000.zip",
    "geonames_admin1CodesASCII.txt": "https://download.geonames.org/export/dump/admin1CodesASCII.txt",
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = RAW / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"downloading {name}")
        req = urllib.request.Request(url, headers={"User-Agent": "MapaniReelPipeline/0.1"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
            f.write(r.read())
    print("building data/geo/layers.gpkg + names.json")
    sys.exit(subprocess.call([sys.executable, "-m", "pipeline.geo.build_layers"], cwd=ROOT))


if __name__ == "__main__":
    main()
