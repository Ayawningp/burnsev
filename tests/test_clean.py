"""Phase 3 test gates: cleaning the pixels (offline).

Two kinds of input:
* tiny hand-built DataArrays on an odc-geo grid (``band`` helper below), for
  the single-function gates;
* Phase 2's ``synthetic_scene`` loaded with ``load_bands``, for ``clean_pair``.
  Read its layout comment in conftest.py: pre nir08/swir22 = 3000/1500 DN,
  post = 1500/3000 DN, SCL a checkerboard of class 4 and class 8 that is the
  same in both dates.

The network is blocked for every test in this file.
"""
import socket
from copy import deepcopy
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr

from burnsev.clean import (
    KEEP,
    PLAUSIBLE,
    CleanPair,
    align,
    clean_pair,
    read_scale_offset,
    scl_mask,
    to_reflectance,
    valid_fraction,
)
from burnsev.crs import CRSMismatchError
from burnsev.load import assert_lazy, load_bands
from burnsev.stac import load_items

REAL = Path(__file__).parent / "fixtures" / "stac" / "palisades_search.json"
SCALE, OFFSET = 0.0001, -0.1  # what sentinel-2-c1-l2a items carry
SO = {"nir08": (SCALE, OFFSET), "swir22": (SCALE, OFFSET)}
ORIGIN = (360000.0, 3770400.0)  # top-left corner, on the 20 m grid
UTM = "EPSG:32611"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def band(arr, res, crs=UTM, chunk=10):
    """Wrap a 2-D numpy array as a lazy DataArray on a north-up grid at ORIGIN."""
    h, w = arr.shape
    gbox = GeoBox((h, w), Affine(res, 0, ORIGIN[0], 0, -res, ORIGIN[1]), crs)
    return wrap_xr(arr, gbox).chunk({"y": chunk, "x": chunk})


def interior(a, k=2):
    """Drop a k-pixel border: resampling edges are not what these gates test."""
    return np.asarray(a)[k:-k, k:-k]


# ====================================================== 1. read_scale_offset

@pytest.mark.parametrize("b", ["nir08", "swir22"])
def test_scale_offset_from_synthetic_item(synthetic_scene, b):
    assert read_scale_offset(synthetic_scene["items"][0], b) == pytest.approx((SCALE, OFFSET))


@pytest.mark.parametrize("b", ["nir08", "swir22"])
def test_scale_offset_from_real_palisades_items(b):
    for item in load_items(REAL):
        scale, offset = read_scale_offset(item, b)
        assert (scale, offset) == pytest.approx((SCALE, OFFSET))
        assert type(scale) is float and type(offset) is float


def test_scl_has_no_scale_offset(synthetic_scene):
    # SCL is a class label: there is nothing to convert.
    with pytest.raises(ValueError, match="scl"):
        read_scale_offset(synthetic_scene["items"][0], "scl")


def test_missing_raster_bands_raises_not_defaults(synthetic_scene):
    item = deepcopy(synthetic_scene["items"][0])
    del item["assets"]["nir08"]["raster:bands"]
    with pytest.raises(ValueError, match="nir08"):
        read_scale_offset(item, "nir08")


def test_missing_offset_raises_not_defaults(synthetic_scene):
    item = deepcopy(synthetic_scene["items"][0])
    del item["assets"]["swir22"]["raster:bands"][0]["offset"]
    with pytest.raises(ValueError, match="swir22"):
        read_scale_offset(item, "swir22")


def test_missing_asset_raises(synthetic_scene):
    with pytest.raises(ValueError, match="B99"):
        read_scale_offset(synthetic_scene["items"][0], "B99")


def test_read_scale_offset_does_not_mutate(synthetic_scene):
    item = synthetic_scene["items"][0]
    before = deepcopy(item)
    read_scale_offset(item, "nir08")
    assert item == before


# ====================================================== 2. to_reflectance

@pytest.fixture
def dn():
    return band(np.array([[0, 1000], [3000, 10000]], dtype="uint16"), 20, chunk=1)


def test_nodata_becomes_nan(dn):
    out = to_reflectance(dn, SCALE, OFFSET, nodata=0).values
    assert np.isnan(out[0, 0])
    assert np.isfinite(out[0, 1]) and np.isfinite(out[1, 0]) and np.isfinite(out[1, 1])


def test_known_dn_converts_to_expected_reflectance(dn):
    # DN 1000 -> 0.0 is the offset gate: offset skipped gives 0.1, applied
    # twice (or in DN units after scaling) gives something else again.
    out = to_reflectance(dn, SCALE, OFFSET, nodata=0).values
    assert out[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert out[1, 0] == pytest.approx(0.2, abs=1e-6)
    assert out[1, 1] == pytest.approx(0.9, abs=1e-6)


def test_reflectance_is_float32(dn):
    assert to_reflectance(dn, SCALE, OFFSET, nodata=0).dtype == np.float32


def test_reflectance_stays_lazy(dn):
    out = to_reflectance(dn, SCALE, OFFSET, nodata=0)
    assert isinstance(out.data, da.Array)


def test_reflectance_keeps_grid(dn):
    out = to_reflectance(dn, SCALE, OFFSET, nodata=0)
    assert out.odc.geobox == dn.odc.geobox


def test_reflectance_does_not_modify_input(dn):
    before = dn.values.copy()
    to_reflectance(dn, SCALE, OFFSET, nodata=0)
    np.testing.assert_array_equal(dn.values, before)
    assert dn.dtype == np.uint16


def test_float_input_is_refused(dn):
    # Already reflectance (or bilinear-resampled DNs): converting again
    # would apply the offset twice.
    once = to_reflectance(dn, SCALE, OFFSET, nodata=0)
    with pytest.raises(TypeError):
        to_reflectance(once, SCALE, OFFSET, nodata=0)


# ====================================================== 3. scl_mask

@pytest.mark.parametrize("cls", range(12))
def test_each_scl_class_kept_or_masked(cls):
    scl = band(np.full((4, 4), cls, dtype="uint8"), 20, chunk=2)
    m = scl_mask(scl).values
    assert m.all() if cls in KEEP else not m.any()


def test_default_keep_is_vegetation_and_bare():
    assert set(KEEP) == {4, 5}


def test_custom_keep_includes_unclassified():
    scl = band(np.arange(12, dtype="uint8").reshape(3, 4), 20, chunk=2)
    m = scl_mask(scl, keep=(4, 5, 7)).values
    np.testing.assert_array_equal(m, np.isin(np.arange(12).reshape(3, 4), [4, 5, 7]))


def test_mask_is_bool_lazy_and_on_same_grid():
    scl = band(np.arange(12, dtype="uint8").reshape(3, 4), 20, chunk=2)
    m = scl_mask(scl)
    assert m.dtype == bool
    assert isinstance(m.data, da.Array)
    assert m.dims == scl.dims
    assert m.odc.geobox == scl.odc.geobox


@pytest.mark.parametrize("keep", [(), (4, 12), (-1,)])
def test_bad_keep_raises(keep):
    scl = band(np.full((2, 2), 4, dtype="uint8"), 20, chunk=1)
    with pytest.raises(ValueError):
        scl_mask(scl, keep=keep)


def test_float_scl_is_refused():
    # Float SCL means someone averaged class labels.
    scl = band(np.full((2, 2), 4.5, dtype="float32"), 20, chunk=1)
    with pytest.raises(TypeError):
        scl_mask(scl)


# ====================================================== 4. align

@pytest.fixture
def mixed_res():
    """10 m NIR, 20 m SWIR and SCL, same 400 m x 400 m footprint."""
    nir = band(np.full((40, 40), 0.2, dtype="float32"), 10)
    swir = band(np.full((20, 20), 0.05, dtype="float32"), 20)
    # Four classes in a repeating pattern; 2x2 blocks guarantee a bilinear
    # resampler would produce in-between values.
    pattern = np.array([4, 5, 8, 9], dtype="uint8")
    idx = (np.add.outer(np.arange(20) // 2, np.arange(20) // 2)) % 4
    scl = band(pattern[idx], 20)
    return nir, swir, scl


@pytest.mark.parametrize("res", [10, 20])
def test_aligned_bands_share_one_grid(mixed_res, res):
    out = align(*mixed_res, target_res=res)
    boxes = [a.odc.geobox for a in out]
    for g in boxes:
        assert g.crs == boxes[0].crs
        assert g.affine == boxes[0].affine  # transform equality, not just shape
        assert g.shape == boxes[0].shape
    assert boxes[0] == boxes[1] == boxes[2]


@pytest.mark.parametrize("res", [10, 20])
def test_aligned_grid_has_target_resolution_and_footprint(mixed_res, res):
    g = align(*mixed_res, target_res=res)[0].odc.geobox
    assert g.resolution.x == res and g.resolution.y == -res
    assert tuple(g.shape) == (400 // res, 400 // res)
    assert (g.affine.c, g.affine.f) == ORIGIN
    assert g.crs.epsg == 32611


@pytest.mark.parametrize("res", [10, 20])
def test_aligned_scl_only_contains_input_classes(mixed_res, res):
    # The nearest-neighbour proof: interpolation would invent 4.5, 6.5, ...
    scl = align(*mixed_res, target_res=res)[2]
    assert np.issubdtype(scl.dtype, np.integer)
    assert set(np.unique(interior(scl.values))) <= {4, 5, 8, 9}


def test_20m_scl_unchanged_when_already_on_target_grid(mixed_res):
    scl_in = mixed_res[2]
    scl_out = align(*mixed_res, target_res=20)[2]
    np.testing.assert_array_equal(scl_out.values, scl_in.values)


@pytest.mark.parametrize("res", [10, 20])
def test_constant_reflectance_survives_resampling(mixed_res, res):
    nir, swir, _ = align(*mixed_res, target_res=res)
    np.testing.assert_allclose(interior(nir.values), 0.2, atol=1e-6)
    np.testing.assert_allclose(interior(swir.values), 0.05, atol=1e-6)


def test_align_stays_lazy(mixed_res):
    for a in align(*mixed_res, target_res=20):
        assert isinstance(a.data, da.Array)


def test_align_refuses_mixed_crs(mixed_res):
    nir, swir, scl = mixed_res
    scl_10n = band(scl.values, 20, crs="EPSG:32610")  # same numbers, other zone
    with pytest.raises(CRSMismatchError):
        align(nir, swir, scl_10n, target_res=20)


# ====================================================== 5. clean_pair

@pytest.fixture(scope="module")
def raw(synthetic_scene):
    ds = load_bands(synthetic_scene["items"], synthetic_scene["aoi"])
    return ds.isel(time=0), ds.isel(time=1)


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean_pair(*raw, SO)


def test_returns_cleanpair_without_scl(cleaned):
    assert isinstance(cleaned, CleanPair)
    for side in (cleaned.pre, cleaned.post):
        assert set(side.data_vars) == {"nir08", "swir22"}


def test_outputs_stay_lazy(cleaned):
    assert_lazy(cleaned.pre)
    assert_lazy(cleaned.post)
    assert isinstance(cleaned.valid.data, da.Array)


def test_outputs_on_input_grid(cleaned, raw):
    g = raw[0].odc.geobox
    assert cleaned.pre.odc.geobox == g
    assert cleaned.post.odc.geobox == g
    assert cleaned.valid.odc.geobox == g


def test_valid_is_2d_bool(cleaned):
    assert cleaned.valid.dims == ("y", "x")
    assert cleaned.valid.dtype == bool


def test_valid_is_clear_in_both_dates(cleaned, raw):
    pre, post = raw
    expected = np.isin(pre.scl.values, KEEP) & np.isin(post.scl.values, KEEP)
    np.testing.assert_array_equal(cleaned.valid.values, expected)
    assert expected.any() and not expected.all()  # fixture sanity: a real mix


def test_same_mask_on_both_dates(cleaned):
    invalid = ~cleaned.valid.values
    for side in (cleaned.pre, cleaned.post):
        for b in ("nir08", "swir22"):
            v = side[b].values
            assert np.isnan(v[invalid]).all(), f"{b} leaks where invalid"
            assert np.isfinite(v[~invalid]).all(), f"{b} lost a valid pixel"


def test_clear_values_are_the_expected_reflectance(cleaned):
    ok = cleaned.valid.values
    expect = {
        "pre": {"nir08": 0.2, "swir22": 0.05},   # 3000, 1500 DN
        "post": {"nir08": 0.05, "swir22": 0.2},  # 1500, 3000 DN
    }
    for name, side in (("pre", cleaned.pre), ("post", cleaned.post)):
        for b, val in expect[name].items():
            np.testing.assert_allclose(side[b].values[ok], val, atol=1e-6)


def test_clear_values_in_plausible_range(cleaned):
    lo, hi = PLAUSIBLE
    ok = cleaned.valid.values
    for side in (cleaned.pre, cleaned.post):
        for b in side.data_vars:
            v = side[b].values[ok]
            assert v.min() >= lo and v.max() <= hi, f"{b}: scaling bug?"


def test_outputs_are_float32(cleaned):
    for side in (cleaned.pre, cleaned.post):
        for b in side.data_vars:
            assert side[b].dtype == np.float32


def test_nodata_in_one_date_invalidates_both(raw):
    pre, post = raw
    pre_np = pre.compute()  # new object; the module fixture stays lazy
    r, c = np.argwhere(pre_np.scl.values == 4)[0]  # a clear-sky pixel
    pre_np["nir08"].data[r, c] = 0  # ...with no measurement in one band
    out = clean_pair(pre_np.chunk(), post, SO)
    assert not bool(out.valid.values[r, c])
    for side in (out.pre, out.post):
        for b in ("nir08", "swir22"):
            assert np.isnan(side[b].values[r, c])


def test_keep_is_passed_through(raw):
    out = clean_pair(*raw, SO, keep=(4, 8))
    assert out.valid.values.all()


def test_mismatched_grids_raise(raw):
    pre, post = raw
    shifted = post.assign_coords(x=post.x + 20)  # one pixel east
    with pytest.raises(ValueError):
        clean_pair(pre, shifted, SO)


def test_missing_scale_offset_band_raises(raw):
    with pytest.raises(ValueError, match="swir22"):
        clean_pair(*raw, {"nir08": (SCALE, OFFSET)})


def test_clean_pair_does_not_modify_inputs(raw):
    pre, post = raw
    clean_pair(pre, post, SO)
    assert pre.nir08.dtype == np.uint16 and "scl" in pre
    assert_lazy(pre)


# ====================================================== 6. valid_fraction

def test_valid_fraction_matches_mask(cleaned):
    f = valid_fraction(cleaned.valid)
    assert type(f) is float
    assert f == pytest.approx(cleaned.valid.values.mean())
    assert 0.4 < f < 0.6  # checkerboard: about half survives


def test_valid_fraction_all_true():
    v = band(np.ones((4, 4), dtype=bool), 20, chunk=2)
    assert valid_fraction(v) == 1.0


def test_valid_fraction_empty_raises():
    with pytest.raises(ValueError):
        valid_fraction(xr.DataArray(np.zeros((0, 0), dtype=bool), dims=("y", "x")))
