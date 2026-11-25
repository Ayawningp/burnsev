"""Phase 2: lazy loading from the cloud.

Turn the STAC items picked in Phase 1 into one lazy xarray Dataset with
dimensions ``(time, y, x)``, on a grid *you* choose: target CRS, resolution
and bounds from the fire perimeter. Nothing is read from disk or the network
until something calls ``.compute()`` (or ``.values``, ``float(...)``, a plot,
a write...).

What is done for you vs. what you write
---------------------------------------
* Done: constants, ``as_pystac_items``, ``describe``.
* YOU DECIDE: the ``RESAMPLING`` table (one method per band, on purpose).
* YOU WRITE: ``load_bands`` (the core task) and ``assert_lazy``.

Call odc-stac as ``odc.stac.load(...)``, not ``from odc.stac import load``:
one test replaces ``odc.stac.load`` with a spy to check which options you
pass, and a name imported directly would dodge the spy.

Pixel values stay raw here: uint16 digital numbers for reflectance, uint8
class codes for SCL, nodata 0. Scaling to reflectance and cloud masking are
Phase 3.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import odc.stac  # noqa: F401  (call it as odc.stac.load; see module docstring)
import pystac
import xarray as xr

from burnsev.crs import assert_crs  # noqa: F401  (you will need it)

import dask.array as da
import pyproj
import odc.geo.geom

TARGET_CRS = "EPSG:32611"  # UTM 11N: the Palisades fire sits well inside it
BANDS = ("nir08", "swir22", "scl")  # B8A, B12, scene classification (all 20 m)
RESOLUTION = 20  # metres; the B8A baseline. 10 m is the B08 variant.
CHUNKS = {"x": 1024, "y": 1024}

# --------------------------------------------------------------------------
# YOU DECIDE: one resampling method per band
# --------------------------------------------------------------------------
# odc-stac resamples whenever the source pixels don't line up 1:1 with your
# output grid: a different resolution (20 m -> 10 m), or a reprojection
# (the AOI spans a neighbouring UTM zone, or you pick another CRS).
#
# Ask of each band: is the number a *measurement* (averaging two neighbours
# gives a sensible in-between value) or a *label* (averaging class 4
# "vegetation" and class 8 "cloud medium probability" gives 6, "water")?
#
# Valid names: "nearest", "bilinear", "cubic", "average", "mode", ...
# (odc-stac passes them to rasterio's Resampling enum).
RESAMPLING: dict[str, str] = {
    "nir08": "bilinear", #Countinuous surface measurement: sensitive to healthy plant cells
    "swir22": "bilinear", #Countinuous surface measurement: sensitive to moisture, soil, and burned surfaces
    "scl": "nearest", #Scene classification layer is an int so we want to keep the int
}


# --------------------------------------------------------------------------
# Helpers (done for you)
# --------------------------------------------------------------------------

def as_pystac_items(items: Sequence[dict]) -> list[pystac.Item]:
    """Convert Phase 1's item dicts to the ``pystac.Item`` objects odc-stac wants.

    The input dicts are copied, never mutated.
    """
    return [pystac.Item.from_dict(it, preserve_dict=True) for it in items]


def describe(ds: xr.Dataset) -> str:
    """One-line summary of a Dataset's size, read from metadata only.

    ``ds.nbytes`` and ``.chunks`` come from shapes and dtypes, so this is safe
    to call on a lazy Dataset: it never triggers a read.
    """
    dims = ", ".join(f"{k}={v}" for k, v in ds.sizes.items())
    chunks = {k: v[0] for k, v in (ds.chunks or {}).items()}
    return (
        f"{len(ds.data_vars)} vars ({', '.join(ds.data_vars)}), {dims}, "
        f"{ds.nbytes / 1e6:.1f} MB if computed, first chunk sizes {chunks}"
    )


# --------------------------------------------------------------------------
# YOU WRITE: guard used by tests now and by later phases
# --------------------------------------------------------------------------

def assert_lazy(ds: xr.Dataset) -> None:
    """Raise ``AssertionError`` naming the culprit if any data variable is not
    backed by a Dask array (i.e. it has already been pulled into memory).

    Hint: a lazy variable's ``.data`` is a ``dask.array.Array``; a computed
    one's is a ``numpy.ndarray``. Look at the variable's ``.data`` attribute,
    never at ``.values`` (that would compute it).
    """
    for name, var in ds.data_vars.items():
        if not isinstance(var.data, da.Array):
            raise AssertionError(
                f"Variable '{name}' is not backed by a Dask array (found"
                f" {type(var.data).__name__})."
            )


# --------------------------------------------------------------------------
# YOU WRITE: Phase 2 core task
# --------------------------------------------------------------------------

def load_bands(
    items: Sequence[dict],
    aoi,
    *,
    bands: Sequence[str] = BANDS,
    crs: str = TARGET_CRS,
    resolution: float = RESOLUTION,
    chunks: Mapping[str, int] = CHUNKS,
    resampling: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Lazily load ``bands`` from ``items``, clipped to the AOI's bounds.

    Parameters
    ----------
    items : sequence of STAC item dicts
        Typically ``pre_items + post_items`` from ``select_scenes``. Several
        tiles from the same day must come back as ONE time step.
    aoi : geopandas.GeoSeries or GeoDataFrame in EPSG:4326
        The same AOI Phase 1 used.
    bands, crs, resolution, chunks
        The output grid and chunking. ``resolution`` is in units of ``crs``.
    resampling : mapping band -> method, optional
        Defaults to the module-level ``RESAMPLING``.

    Rules (the tests check each one)
    --------------------------------
    1. ``items`` empty -> ``ValueError``.
    2. Assert the AOI is EPSG:4326 with ``assert_crs`` before using it.
    3. ``crs`` must be a projected CRS (metres), else ``ValueError``: in
       EPSG:4326 a "pixel" is a different size on the ground at every
       latitude and your Phase 4 areas would be wrong.
       Hint: ``pyproj.CRS.from_user_input(crs).is_projected``.
    4. Every requested band needs an explicit resampling method; a missing
       one -> ``ValueError`` naming it. Never let odc-stac fall back to its
       default silently.
    5. Call ``odc.stac.load`` with ``bands``, ``crs``, ``resolution``,
       ``chunks``, ``resampling``, ``groupby="solar_day"`` and the AOI as
       ``geopolygon``. The output grid then covers the AOI and little more.
    6. Return the Dataset with every data variable still a Dask array
       (finish with ``assert_lazy``). Do not touch ``.values`` anywhere.

    Hints
    -----
    * ``as_pystac_items(items)`` converts the dicts.
    * ``geopolygon`` wants a geometry that carries its CRS. A bare shapely
      polygon has none, so wrap it:
      ``odc.geo.geom.Geometry(aoi.union_all(), crs=...)``. Why does odc-stac
      need the AOI's CRS when you already told it the output ``crs``?
    * Why ``groupby="solar_day"``? Look at the two 2024-12-28 tiles in the
      test fixture: same overpass, timestamps 20 s apart. With the default
      grouping, what would ``ds.sizes["time"]`` be?
    """

    #Make sure items aren't empty
    if not items:
        raise ValueError("No STAC items to load: items is empty.")

    #Validate CRS
    assert_crs(aoi, "EPSG:4326")

    #Output CRS must be projected (metres)
    if not pyproj.CRS.from_user_input(crs).is_projected:
        raise ValueError(f"Output crs must be a projected CRS, got {crs!r}.")

    #Every requested band needs a specific respampling method
    resampling_map = RESAMPLING if resampling is None else resampling
    for band in bands:
        if band not in resampling_map:
            raise ValueError(f"Missing resampling method for band '{band}'.")

    #Wrap AOI geometry with its CRS and load lazily via odc-stac
    geopolygon = odc.geo.geom.Geometry(aoi.union_all(), crs=aoi.crs) #odc-stac doesn't know degres or metres
    ds = odc.stac.load(
      as_pystac_items(items),
      bands=list(bands),
      crs=crs,
      resolution=resolution,
      chunks=dict(chunks),
      resampling=dict(resampling_map),
      groupby="solar_day", #groupby="time" groups by exact timestamp
      geopolygon=geopolygon,
    )

    #Check that dataset is still lazy dask array (assert_lazy)
    assert_lazy(ds)
    return ds
