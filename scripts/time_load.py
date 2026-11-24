"""Time a lazy load against real Sentinel-2 COGs (Phase 2 tasks 2-3). LIVE: needs network.

Run from the repo root once load_bands passes its tests:

    python scripts/time_load.py

Uses the scenes Phase 1 picks from the saved Palisades search, so the STAC
search itself is not repeated; only the pixel reads go over the network.
Prints the Dataset's size *before* computing, then times one small window
and the whole AOI. Note the numbers in issue #3.
"""
import time
from datetime import date
from pathlib import Path

import geopandas as gpd
import odc.stac
from shapely.geometry import box

from burnsev.load import assert_lazy, describe, load_bands
from burnsev.stac import load_items, select_scenes

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "stac" / "palisades_search.json"
AOI_BOUNDS = (-118.68, 34.03, -118.50, 34.13)  # keep in sync with save_stac_fixture.py
FIRE_START, FIRE_END = date(2025, 1, 7), date(2025, 1, 31)
WINDOW = 256  # pixels per side for the small-window timing


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"{label}: {time.perf_counter() - t0:.1f} s")
    return out


if __name__ == "__main__":
    # GDAL settings for reading COGs over HTTP: anonymous S3 access, no
    # directory listing on open, merged range requests. Without these every
    # open makes extra round-trips.
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

    aoi = gpd.GeoSeries([box(*AOI_BOUNDS)], crs="EPSG:4326")
    pre, post = select_scenes(load_items(FIXTURE), aoi, FIRE_START, FIRE_END)
    print("pre :", [it["id"] for it in pre])
    print("post:", [it["id"] for it in post])

    ds = timed("build lazy Dataset", lambda: load_bands(pre + post, aoi))
    assert_lazy(ds)
    print("before compute:", describe(ds))

    window = ds.isel(x=slice(0, WINDOW), y=slice(0, WINDOW))
    timed(f"compute {WINDOW}x{WINDOW} window", window.compute)
    full = timed("compute whole AOI", ds.compute)
    print("after compute :", describe(full))
