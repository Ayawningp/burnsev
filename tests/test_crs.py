"""Phase 0 gates for burnsev.crs.

Interface these tests expect from src/burnsev/crs.py:

    class CRSMismatchError(ValueError)
    utm_epsg(lon, lat) -> int
    assert_crs(obj, expected) -> None
    pixel_to_map(transform, row, col) -> (x, y)      # pixel CENTER
    map_to_pixel(transform, x, y) -> (row, col)      # ints, the pixel containing (x, y)
"""
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import geopandas as gpd
import pytest
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.transform import from_origin, rowcol, xy
from shapely.geometry import Point, box

LA = (-118.24, 34.05)


# ---------------------------------------------------------------- packaging


def test_import_has_no_side_effects(tmp_path):
    """Importing the library must not print, write files, or do work."""
    result = subprocess.run(
        [sys.executable, "-c", "import burnsev.crs"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", f"import printed: {result.stdout!r}"
    assert list(tmp_path.iterdir()) == [], "import created files in cwd"


# ---------------------------------------------------------------- utm_epsg


@pytest.mark.parametrize(
    "lon, lat, expected",
    [
        (-118.24, 34.05, 32611),  # Los Angeles (README gate)
        (151.21, -33.87, 32756),  # Sydney: southern hemisphere -> 327xx
        (-180.0, 10.0, 32601),  # western edge -> zone 1
        (180.0, 10.0, 32660),  # eastern edge must not become zone 61
        (-117.0, 0.0, 32611),  # equator counts as north
        (0.0, -0.0001, 32731),  # just south of the equator
    ],
)
def test_utm_epsg_known_points(crs_mod, lon, lat, expected):
    assert crs_mod.utm_epsg(lon, lat) == expected


@pytest.mark.parametrize("lon, lat", [(0, 85), (0, -81), (181, 0), (-181, 0)])
def test_utm_epsg_rejects_out_of_range(crs_mod, lon, lat):
    with pytest.raises(ValueError):
        crs_mod.utm_epsg(lon, lat)


def test_utm_epsg_catches_swapped_lat_lon(crs_mod):
    """Trap from the README: passing (lat, lon) must fail loudly, not silently."""
    lat, lon = LA[1], LA[0]
    with pytest.raises(ValueError):
        crs_mod.utm_epsg(lat, lon)  # wrong order on purpose: lat=-118 is impossible


# ---------------------------------------------------------------- area


def test_one_km_square_in_utm_has_one_km2_area():
    """README gate: 1 km x 1 km built in UTM -> 1,000,000 m^2 within 0.1%."""
    sq = gpd.GeoSeries([box(400_000, 3_700_000, 401_000, 3_701_000)], crs="EPSG:32611")
    assert sq.area.iloc[0] == pytest.approx(1_000_000, rel=1e-3)


def test_utm_area_close_to_equal_area_but_slightly_smaller(crs_mod):
    """0.1 deg box west of zone 11's central meridian (-117).

    UTM's scale factor there is < 1, so UTM area should be a hair smaller than
    the equal-area answer, but within 0.1%.
    """
    poly = gpd.GeoSeries([box(-118.34, 34.05, -118.24, 34.15)], crs="EPSG:4326")
    c = poly.iloc[0].centroid
    utm = poly.to_crs(epsg=crs_mod.utm_epsg(c.x, c.y)).area.iloc[0]
    ea = poly.to_crs("EPSG:6933").area.iloc[0]
    assert utm == pytest.approx(ea, rel=1e-3)
    assert utm < ea


# ---------------------------------------------------------------- pixel <-> map

NORTH_UP = from_origin(420000.0, 3730000.0, 10.0, 10.0)
# Rotated/sheared grid: exercises the b and d terms of the affine.
ROTATED = Affine(10.0, 2.0, 420000.0, 1.5, -10.0, 3730000.0)


@pytest.mark.parametrize("T", [NORTH_UP, ROTATED], ids=["north_up", "rotated"])
@pytest.mark.parametrize("row, col", [(0, 0), (15, 42), (99, 7)])
def test_pixel_to_map_matches_rasterio(crs_mod, T, row, col):
    x, y = crs_mod.pixel_to_map(T, row, col)
    x_ref, y_ref = xy(T, row, col, offset="center")
    assert x == pytest.approx(x_ref)
    assert y == pytest.approx(y_ref)


@pytest.mark.parametrize("T", [NORTH_UP, ROTATED], ids=["north_up", "rotated"])
@pytest.mark.parametrize("row, col", [(0, 0), (15, 42), (99, 7)])
def test_pixel_map_pixel_round_trip(crs_mod, T, row, col):
    """README gate: pixel -> map -> pixel returns the same (row, col)."""
    x, y = crs_mod.pixel_to_map(T, row, col)
    assert crs_mod.map_to_pixel(T, x, y) == (row, col)


def test_map_to_pixel_off_center_point(crs_mod):
    """A point inside the pixel but away from its center still maps to that pixel."""
    T = NORTH_UP
    # 9 m east, 1 m south of the top-left corner of pixel (15, 42)
    x = T.c + 42 * T.a + 9.0
    y = T.f + 15 * T.e - 1.0
    assert crs_mod.map_to_pixel(T, x, y) == (15, 42)
    assert crs_mod.map_to_pixel(T, x, y) == tuple(rowcol(T, x, y))


def test_map_to_pixel_returns_ints(crs_mod):
    r, c = crs_mod.map_to_pixel(NORTH_UP, 420_105.0, 3_729_895.0)
    assert isinstance(r, int) and isinstance(c, int)


# ---------------------------------------------------------------- assert_crs


def test_mismatch_error_is_a_value_error(crs_mod):
    assert issubclass(crs_mod.CRSMismatchError, ValueError)


def test_assert_crs_passes_on_match_gdf(crs_mod):
    gdf = gpd.GeoDataFrame(geometry=[Point(*LA)], crs="EPSG:4326")
    assert crs_mod.assert_crs(gdf, "EPSG:4326") is None
    assert crs_mod.assert_crs(gdf, 4326) is None
    assert crs_mod.assert_crs(gdf, CRS.from_epsg(4326)) is None


def test_assert_crs_raises_on_mismatch_gdf(crs_mod):
    """README gate: assert_crs raises on a mismatch."""
    gdf = gpd.GeoDataFrame(geometry=[Point(*LA)], crs="EPSG:4326")
    with pytest.raises(crs_mod.CRSMismatchError) as exc:
        crs_mod.assert_crs(gdf, "EPSG:32611")
    msg = str(exc.value)
    assert "4326" in msg and "32611" in msg


def test_assert_crs_geoseries(crs_mod):
    gs = gpd.GeoSeries([Point(*LA)], crs="EPSG:4326")
    crs_mod.assert_crs(gs, "EPSG:4326")
    with pytest.raises(crs_mod.CRSMismatchError):
        crs_mod.assert_crs(gs, "EPSG:32611")


def test_assert_crs_set_crs_is_not_reprojection(crs_mod):
    """Trap: set_crs only relabels. The label matches, so assert_crs passes --
    this documents that assert_crs checks labels, not whether coords make sense."""
    gdf = gpd.GeoDataFrame(geometry=[Point(*LA)], crs="EPSG:4326")
    mislabeled = gdf.set_crs("EPSG:32611", allow_override=True)
    crs_mod.assert_crs(mislabeled, "EPSG:32611")
    assert mislabeled.geometry.iloc[0].x == pytest.approx(LA[0])  # still degrees!


def test_assert_crs_raises_when_gdf_has_no_crs(crs_mod):
    gdf = gpd.GeoDataFrame(geometry=[Point(*LA)])  # crs=None
    with pytest.raises(crs_mod.CRSMismatchError, match="(?i)no crs"):
        crs_mod.assert_crs(gdf, "EPSG:4326")


def test_assert_crs_rasterio_dataset(crs_mod, make_geotiff):
    path = make_geotiff(crs="EPSG:32611")
    with rasterio.open(path) as src:
        crs_mod.assert_crs(src, "EPSG:32611")
        with pytest.raises(crs_mod.CRSMismatchError):
            crs_mod.assert_crs(src, "EPSG:32610")


def test_assert_crs_rasterio_dataset_without_crs(crs_mod, make_geotiff):
    path = make_geotiff(crs=None)
    with rasterio.open(path) as src:
        with pytest.raises(crs_mod.CRSMismatchError, match="(?i)no crs"):
            crs_mod.assert_crs(src, "EPSG:32611")


def test_assert_crs_accepts_crs_like_inputs(crs_mod):
    crs_mod.assert_crs(CRS.from_epsg(32611), "EPSG:32611")
    crs_mod.assert_crs(rasterio.crs.CRS.from_epsg(32611), "EPSG:32611")
    crs_mod.assert_crs("EPSG:32611", 32611)


def test_assert_crs_rejects_bare_shapely_geometry(crs_mod):
    """Shapely geometries carry no CRS: that's a usage error, not a mismatch."""
    with pytest.raises(TypeError, match="(?i)shapely"):
        crs_mod.assert_crs(Point(*LA), "EPSG:4326")


def test_assert_crs_rejects_none_expected(crs_mod):
    gdf = gpd.GeoDataFrame(geometry=[Point(*LA)], crs="EPSG:4326")
    with pytest.raises(ValueError):
        crs_mod.assert_crs(gdf, None)


# xarray objects (Phases 2-4) keep their CRS behind an accessor, not at .crs.
# These stand-ins mimic that shape so the test needs no rioxarray / odc-stac.


def test_assert_crs_rioxarray_style_accessor(crs_mod):
    da = SimpleNamespace(rio=SimpleNamespace(crs=CRS.from_epsg(32611)))
    crs_mod.assert_crs(da, "EPSG:32611")
    with pytest.raises(crs_mod.CRSMismatchError):
        crs_mod.assert_crs(da, "EPSG:4326")


def test_assert_crs_odc_style_accessor(crs_mod):
    ds = SimpleNamespace(odc=SimpleNamespace(crs=CRS.from_epsg(32611)))
    crs_mod.assert_crs(ds, "EPSG:32611")
    with pytest.raises(crs_mod.CRSMismatchError):
        crs_mod.assert_crs(ds, "EPSG:4326")


def test_assert_crs_accessor_with_no_crs(crs_mod):
    da = SimpleNamespace(rio=SimpleNamespace(crs=None))
    with pytest.raises(crs_mod.CRSMismatchError, match="(?i)no crs"):
        crs_mod.assert_crs(da, "EPSG:32611")


def test_assert_crs_still_raises_under_python_O(tmp_path):
    """A bare `assert` is stripped by `python -O`; the gate must not be."""
    code = textwrap.dedent(
        """
        import geopandas as gpd
        from shapely.geometry import Point
        from burnsev.crs import assert_crs
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
        assert_crs(gdf, "EPSG:32611")
        """
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode != 0, "mismatch passed silently under -O"
    assert "CRSMismatchError" in result.stderr
