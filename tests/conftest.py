"""Shared offline fixtures for the burnsev test suite."""
import importlib
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

NODATA = -9999.0


@pytest.fixture(scope="session")
def crs_mod(tmp_path_factory):
    """Import burnsev.crs from inside a throwaway directory.

    If the module still has import-time side effects (writing files),
    they land in a temp dir instead of the repo. test_import_has_no_side_effects
    is the test that actually catches them.
    """
    old_cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("import_sandbox"))
    try:
        return importlib.import_module("burnsev.crs")
    finally:
        os.chdir(old_cwd)


@pytest.fixture
def make_geotiff(tmp_path):
    """Factory: write a tiny float32 GeoTIFF and return its path.

    Pass crs=None to write a raster with no CRS at all.
    """

    def _make(crs="EPSG:32611", transform=None, shape=(20, 30), name="tiny.tif"):
        if transform is None:
            transform = from_origin(420000.0, 3730000.0, 10.0, 10.0)
        data = np.arange(shape[0] * shape[1], dtype="float32").reshape(shape)
        data[0, 0] = NODATA  # one explicit nodata pixel
        path = tmp_path / name
        profile = dict(
            driver="GTiff",
            height=shape[0],
            width=shape[1],
            count=1,
            dtype="float32",
            transform=transform,
            nodata=NODATA,
        )
        if crs is not None:
            profile["crs"] = crs
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
        return path

    return _make


# --------------------------------------------------------------------------
# Phase 2: tiny synthetic COGs + STAC items that point at them on disk
# --------------------------------------------------------------------------
#
# Layout (all in EPSG:32611, 20 m pixels on the global 20 m grid):
#
#   y 3771000 +--------------------+
#             | WEST tile          |
#             |  x 359000..360600  |  EAST tile x 360400..362000 overlaps
#             |       +-AOI-+      |  the west one by 200 m; the AOI
#             |       |     |      |  straddles the seam, so only the two
#   y 3769000 +-------+-----+------+  tiles together cover it.
#
#   AOI: x 360005..360995, y 3770005..3770595 (deliberately off-grid),
#   handed to the code in EPSG:4326 like Phase 1's AOI.
#
# Two dates, two tiles each, 20 s apart (same overpass):
#   2024-12-28  pre:  nir08 = 3000, swir22 = 1500
#   2025-02-06  post: nir08 = 1500, swir22 = 3000   (burned: NIR down, SWIR up)
# SCL is a checkerboard of class 4 (vegetation) and 8 (cloud medium prob.),
# keyed to absolute pixel position so the tiles agree in the overlap. Any
# averaging resampler turns it into values that are not 4 or 8.

S2_EPSG = 32611
S2_RES = 20
SYN_TILES = {"WEST": (359000, 3769000, 360600, 3771000), "EAST": (360400, 3769000, 362000, 3771000)}
SYN_AOI_UTM = (360005.0, 3770005.0, 360995.0, 3770595.0)
SYN_DATES = {
    "2024-12-28": {"nir08": 3000, "swir22": 1500},
    "2025-02-06": {"nir08": 1500, "swir22": 3000},
}
SCL_CLASSES = (4, 8)
COG_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"


def _scl_checkerboard(bounds):
    minx, _, maxx, maxy = bounds
    w = int((maxx - minx) // S2_RES)
    h = int((maxy - bounds[1]) // S2_RES)
    col0, row0 = int(minx // S2_RES), int(-maxy // S2_RES)
    rows, cols = np.indices((h, w))
    parity = (rows + row0 + cols + col0) % 2
    return np.where(parity == 0, SCL_CLASSES[0], SCL_CLASSES[1]).astype("uint8")


def _write_cog(path, data, bounds):
    h, w = data.shape
    with rasterio.open(
        path, "w", driver="COG", height=h, width=w, count=1, dtype=str(data.dtype),
        crs=f"EPSG:{S2_EPSG}", transform=from_origin(bounds[0], bounds[3], S2_RES, S2_RES),
        nodata=0,
    ) as dst:
        dst.write(data, 1)


def _footprint_4326(bounds):
    from pyproj import Transformer

    to_ll = Transformer.from_crs(S2_EPSG, 4326, always_xy=True)
    minx, miny, maxx, maxy = bounds
    ring = [to_ll.transform(x, y) for x, y in ((minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy))]
    ring.append(ring[0])
    lons, lats = zip(*ring)
    return {"type": "Polygon", "coordinates": [[list(p) for p in ring]]}, [min(lons), min(lats), max(lons), max(lats)]


def _stac_item(item_id, when, bounds, hrefs):
    minx, miny, maxx, maxy = bounds
    shape_ = [int((maxy - miny) // S2_RES), int((maxx - minx) // S2_RES)]
    geometry, bbox = _footprint_4326(bounds)
    assets = {}
    for band, href in hrefs.items():
        rb = {"nodata": 0, "data_type": "uint8" if band == "scl" else "uint16", "spatial_resolution": S2_RES}
        if band != "scl":
            rb.update(scale=0.0001, offset=-0.1)  # as in real sentinel-2-c1-l2a items
        assets[band] = {
            "href": str(href), "type": COG_TYPE, "roles": ["data"],
            "proj:shape": shape_, "proj:transform": [S2_RES, 0, minx, 0, -S2_RES, maxy],
            "raster:bands": [rb],
        }
    return {
        "type": "Feature", "stac_version": "1.0.0", "id": item_id,
        "stac_extensions": [
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
        ],
        "geometry": geometry, "bbox": bbox,
        "properties": {"datetime": when, "proj:epsg": S2_EPSG, "eo:cloud_cover": 0.0},
        "links": [], "assets": assets,
    }


@pytest.fixture(scope="session")
def synthetic_scene(tmp_path_factory):
    """Write the COGs once per session; return items, AOIs and expected values."""
    import geopandas as gpd
    from shapely.geometry import box

    root = tmp_path_factory.mktemp("cogs")
    items = []
    for day, refl in SYN_DATES.items():
        for k, (tile, bounds) in enumerate(SYN_TILES.items()):
            w = int((bounds[2] - bounds[0]) // S2_RES)
            h = int((bounds[3] - bounds[1]) // S2_RES)
            hrefs = {}
            for band in ("nir08", "swir22", "scl"):
                data = _scl_checkerboard(bounds) if band == "scl" else np.full((h, w), refl[band], "uint16")
                hrefs[band] = root / f"{tile}_{day}_{band}.tif"
                _write_cog(hrefs[band], data, bounds)
            when = f"{day}T18:45:{k * 20:02d}Z"  # same overpass, 20 s apart
            items.append(_stac_item(f"SYN_{tile}_{day.replace('-', '')}", when, bounds, hrefs))

    aoi_utm = gpd.GeoSeries([box(*SYN_AOI_UTM)], crs=S2_EPSG)
    return {
        "items": items,
        "aoi": aoi_utm.to_crs(4326),  # what the code receives
        "aoi_utm": aoi_utm,           # what the tests measure against
        "dates": list(SYN_DATES),
        "values": SYN_DATES,
        "missing_dir": root / "does_not_exist",
    }
