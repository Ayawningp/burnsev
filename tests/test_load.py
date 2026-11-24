"""Phase 2 test gates: lazy loading (offline).

Fixture: ``synthetic_scene`` in conftest.py writes tiny real COGs to a temp
dir and builds STAC items pointing at them; read its layout comment first.
The real-items test reuses Phase 1's saved Palisades search and never
computes, so it runs offline too.

The network is blocked for every test in this file.
"""
import socket
from copy import deepcopy
from datetime import date
from pathlib import Path

import dask.array as da
import numpy as np
import odc.stac
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

import burnsev.load as load_mod
from burnsev.crs import CRSMismatchError, assert_crs
from burnsev.load import BANDS, RESAMPLING, TARGET_CRS, assert_lazy, describe, load_bands
from burnsev.stac import load_items, select_scenes

REAL = Path(__file__).parent / "fixtures" / "stac" / "palisades_search.json"
REAL_AOI_BOUNDS = (-118.68, 34.03, -118.50, 34.13)
FIRE_START, FIRE_END = date(2025, 1, 7), date(2025, 1, 31)

ODC_RESAMPLERS = {"nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "min", "max", "med", "q1", "q3"}
CATEGORICAL_OK = {"nearest", "mode"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(scope="module")
def ds(synthetic_scene):
    return load_bands(synthetic_scene["items"], synthetic_scene["aoi"])


def grid_box(ds):
    return box(*ds.odc.geobox.boundingbox)


# ---------------------------------------------------------------- resampling choice

def test_resampling_names_every_band():
    assert set(RESAMPLING) == set(BANDS)


def test_resampling_methods_are_real():
    assert set(RESAMPLING.values()) <= ODC_RESAMPLERS


def test_scl_is_never_averaged():
    # SCL holds class labels: any interpolating method invents classes.
    assert RESAMPLING["scl"] in CATEGORICAL_OK


# ---------------------------------------------------------------- laziness

def test_returns_dataset_with_bands_and_dims(ds):
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == set(BANDS)
    for v in BANDS:
        assert ds[v].dims == ("time", "y", "x")


def test_every_variable_is_dask(ds):
    for v in BANDS:
        assert isinstance(ds[v].data, da.Array), f"{v} was computed at load time"


def test_load_reads_no_pixels(synthetic_scene):
    # Point every asset at a file that does not exist. A lazy load still
    # succeeds, because only STAC metadata is needed to build the grid;
    # the missing files only matter once something computes.
    broken = deepcopy(synthetic_scene["items"])
    for it in broken:
        for band, asset in it["assets"].items():
            asset["href"] = str(synthetic_scene["missing_dir"] / f"{it['id']}_{band}.tif")
    lazy = load_bands(broken, synthetic_scene["aoi"])
    describe(lazy)  # metadata only: must not trigger a read either
    with pytest.raises(Exception):
        lazy["nir08"].isel(time=0).compute()


def test_assert_lazy_passes_on_lazy_and_fails_once_computed(ds):
    assert_lazy(ds)
    with pytest.raises(AssertionError, match="nir08|swir22|scl"):
        assert_lazy(ds.compute())


# ---------------------------------------------------------------- grid

def test_crs_is_target(ds):
    assert_crs(ds, TARGET_CRS)  # goes through ds.odc.crs


def test_resolution_is_20m(ds):
    res = ds.odc.geobox.resolution
    assert (abs(res.x), abs(res.y)) == (20, 20)


def test_grid_covers_aoi(ds, synthetic_scene):
    assert grid_box(ds).covers(synthetic_scene["aoi_utm"].union_all())


def test_grid_is_tight_around_aoi(ds, synthetic_scene):
    # Loading the AOI, not the tile: at most one pixel of slack on each side.
    slack = box(*synthetic_scene["aoi_utm"].total_bounds).buffer(20, join_style="mitre")
    assert slack.covers(grid_box(ds))


# ---------------------------------------------------------------- time

def test_same_day_tiles_become_one_time_step(ds):
    assert ds.sizes["time"] == 2


def test_time_steps_are_the_pre_and_post_dates(ds, synthetic_scene):
    days = [d.isoformat() for d in pd.DatetimeIndex(ds.time.values).date]
    assert days == synthetic_scene["dates"]


# ---------------------------------------------------------------- pixels (computes the tiny fixture)

def test_values_mosaic_across_the_tile_seam(ds, synthetic_scene):
    # The AOI straddles two tiles; a seam or a missing tile would show up as
    # nodata (0) or as the wrong date's value.
    for t, day in enumerate(synthetic_scene["dates"]):
        for band, expected in synthetic_scene["values"][day].items():
            arr = ds[band].isel(time=t).values
            assert (arr == expected).all(), f"{band} on {day}"
        scl = ds["scl"].isel(time=t).values
        assert set(np.unique(scl)) == {4, 8}


def test_raw_dtypes_and_nodata_kept(ds):
    # Phase 2 does not scale or mask: that is Phase 3's job, done on purpose.
    assert ds["nir08"].dtype == np.uint16
    assert ds["swir22"].dtype == np.uint16
    assert ds["scl"].dtype == np.uint8
    for v in BANDS:
        assert ds[v].attrs.get("nodata") == 0


def test_10m_grid_keeps_scl_categorical(synthetic_scene):
    ds10 = load_bands(synthetic_scene["items"], synthetic_scene["aoi"], resolution=10)
    assert abs(ds10.odc.geobox.resolution.x) == 10
    scl = ds10["scl"].values
    assert set(np.unique(scl)) <= {4, 8}, "SCL was interpolated into made-up classes"
    assert (ds10["nir08"].isel(time=0).values == 3000).all()


# ---------------------------------------------------------------- input checks

def test_rejects_empty_items(synthetic_scene):
    with pytest.raises(ValueError):
        load_bands([], synthetic_scene["aoi"])


def test_rejects_geographic_crs(synthetic_scene):
    with pytest.raises(ValueError):
        load_bands(synthetic_scene["items"], synthetic_scene["aoi"], crs="EPSG:4326", resolution=0.0002)


def test_rejects_aoi_not_in_4326(synthetic_scene):
    with pytest.raises(CRSMismatchError):
        load_bands(synthetic_scene["items"], synthetic_scene["aoi_utm"])


def test_rejects_aoi_without_crs(synthetic_scene):
    bare = synthetic_scene["aoi"].set_crs(None, allow_override=True)
    with pytest.raises(CRSMismatchError):
        load_bands(synthetic_scene["items"], bare)


def test_rejects_band_without_resampling(synthetic_scene):
    partial = {"nir08": "bilinear", "swir22": "bilinear"}
    with pytest.raises(ValueError, match="scl"):
        load_bands(synthetic_scene["items"], synthetic_scene["aoi"], resampling=partial)


def test_does_not_mutate_items(synthetic_scene):
    before = deepcopy(synthetic_scene["items"])
    load_bands(synthetic_scene["items"], synthetic_scene["aoi"])
    assert synthetic_scene["items"] == before


# ---------------------------------------------------------------- options reach odc-stac

def test_passes_explicit_grid_and_resampling_to_odc(synthetic_scene, monkeypatch):
    seen = {}
    real_load = odc.stac.load

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(odc.stac, "load", spy)
    load_bands(synthetic_scene["items"], synthetic_scene["aoi"])

    assert seen, "odc.stac.load was not called (call it as odc.stac.load)"
    assert seen.get("crs") == TARGET_CRS
    assert seen.get("resolution") == 20
    assert seen.get("chunks") == load_mod.CHUNKS
    assert seen.get("groupby") == "solar_day"
    assert "geopolygon" in seen
    assert isinstance(seen.get("resampling"), dict)
    assert set(seen["resampling"]) >= set(BANDS)


# ---------------------------------------------------------------- real items, metadata only

@pytest.mark.skipif(not REAL.exists(), reason="no saved Palisades search")
def test_real_items_load_lazily_offline():
    import geopandas as gpd

    aoi = gpd.GeoSeries([box(*REAL_AOI_BOUNDS)], crs="EPSG:4326")
    pre, post = select_scenes(load_items(REAL), aoi, FIRE_START, FIRE_END)
    lazy = load_bands(pre + post, aoi)  # network is blocked: this must not read

    assert_lazy(lazy)
    assert_crs(lazy, TARGET_CRS)
    assert lazy.sizes["time"] == 2
    assert grid_box(lazy).covers(aoi.to_crs(TARGET_CRS).union_all())
    # ~16 x 11 km at 20 m: on the order of 800 x 550 pixels, not 5490 x 5490.
    assert lazy.sizes["x"] < 1200 and lazy.sizes["y"] < 1200
