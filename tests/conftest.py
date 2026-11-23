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
