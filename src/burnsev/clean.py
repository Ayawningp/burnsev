"""Phase 3: cleaning the pixels.

Turn Phase 2's raw lazy Dataset (uint16 digital numbers, uint8 SCL codes,
nodata 0) into float surface reflectance with clouds, shadows and missing
data set to NaN, every band on one grid, and a single validity mask shared
by the pre- and post-fire images.

Order of operations (the tests check the result, not the route):
    nodata -> scale/offset -> SCL mask -> (resample, only if grids differ)

What is done for you vs. what you write
---------------------------------------
* Done: constants, ``CleanPair``.
* YOU WRITE: ``read_scale_offset``, ``to_reflectance``, ``scl_mask``,
  ``align``, ``clean_pair``, ``valid_fraction``.

Everything except ``valid_fraction`` must stay lazy: Dask in, Dask out. Never
call ``.values``, ``.compute()`` or ``.load()`` here.
"""
from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

import numpy as np
import xarray as xr

import odc.geo.xr  # Registers the .odc accessor on xarray objects
from odc.geo.geobox import GeoBox

from burnsev.crs import assert_crs  # noqa: F401  (you will need it)
from burnsev.load import assert_lazy  # noqa: F401  (you will need it)

import rasterio

# SCL classes kept by default: 4 vegetation, 5 not-vegetated (bare / burned).
# 7 "unclassified" is your call; the roadmap asks you to try both.
KEEP = (4, 5)
SCL_CLASSES = range(12)  # 0..11 are the only defined SCL values
NODATA = 0  # L2A fill value for every band, SCL included
PLAUSIBLE = (-0.1, 1.2)  # clear-sky reflectance far outside this = scaling bug


class CleanPair(NamedTuple):
    """Output of ``clean_pair``.

    pre, post : xr.Dataset
        Reflectance bands only (no ``scl``), float32, NaN wherever
        ``valid`` is False. Same grid as the inputs.
    valid : xr.DataArray of bool, dims (y, x)
        True only where the pixel is clear and has data in BOTH dates.
    """

    pre: xr.Dataset
    post: xr.Dataset
    valid: xr.DataArray


# --------------------------------------------------------------------------
# YOU WRITE: 1. reading the conversion from metadata
# --------------------------------------------------------------------------

def read_scale_offset(item: dict, band: str) -> tuple[float, float]:
    """Return ``(scale, offset)`` for ``band`` from a STAC item dict.

    They live at ``item["assets"][band]["raster:bands"][0]``.

    Rules
    -----
    * Missing asset, missing ``raster:bands``, or missing ``scale`` or
      ``offset`` -> ``ValueError`` naming the band. Do not fall back to
      1 and 0: a silent default is how an offset gets skipped.
    * Return plain Python floats.
    """
    try:
        assets = item["assets"][band]["raster:bands"][0]
        scale = assets["scale"]
        offset = assets["offset"]
        if scale is None or offset is None:
            raise TypeError("scale or offset is None")
        return float(scale), float(offset)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing or invalid scale/offset metadata for band '{band}'"
        ) from exc

# --------------------------------------------------------------------------
# YOU WRITE: 2. digital numbers -> reflectance
# --------------------------------------------------------------------------

def to_reflectance(
    da: xr.DataArray, scale: float, offset: float, nodata: int | float
) -> xr.DataArray:
    """Convert raw digital numbers to surface reflectance.

    Rules
    -----
    * ``da`` must have an integer dtype, else ``TypeError``. Float input
      means it was already converted (or resampled with an interpolating
      method); converting again would apply the offset twice.
    * Pixels equal to ``nodata`` become NaN.
    * Everything else becomes ``DN * scale + offset`` (STAC convention).
    * Output dtype float32; coordinates (and so the grid) unchanged.
    * Lazy in, lazy out. ``da`` itself is not modified.
    """

#Intution: Trading decimal precision for storage efficiency while preventing data clipping at zero.
#If a pixel in an HLS image has a DN of 3,500:
#This means the surface at that pixel reflects 25% of the light in that spectral band.

    if not np.issubdtype(da.dtype, np.integer):
        raise TypeError(
                    f"Expected integer dtype for raw DN DataArray, got '{da.dtype}'."
        )
    masked = da.where(da != nodata)
    reflectance = (masked * scale + offset).astype("float32")
    return reflectance


# --------------------------------------------------------------------------
# YOU WRITE: 3. cloud / shadow mask
# --------------------------------------------------------------------------

def scl_mask(scl: xr.DataArray, keep: Sequence[int] = KEEP) -> xr.DataArray:
    """Boolean mask, True where the SCL class is in ``keep``.

    Rules
    -----
    * ``keep`` empty, or containing a value outside 0..11 -> ``ValueError``.
    * ``scl`` must have an integer dtype, else ``TypeError`` (float SCL means
      something averaged class labels).
    * Returns dtype bool, same dims and coords as ``scl``. Lazy in, lazy out.
    """
    keep_list = list(keep)
    if not keep_list or any(
        isinstance(c, bool) or not isinstance(c, int) or c not in SCL_CLASSES
        for c in keep_list
    ):
        raise ValueError(
            f"'keep' must be a non-empty sequence of integers in 0..11, got {keep_list!r}."
        )

    if not np.issubdtype(scl.dtype, np.integer):
        raise TypeError(
            f"Expected integer dtype for SCL DataArray, got '{scl.dtype}'."
        )

    return scl.isin(keep_list)


# --------------------------------------------------------------------------
# YOU WRITE: 4. one grid for every band
# --------------------------------------------------------------------------

def _reflectance_resampling(band: xr.DataArray, target_res: float) -> str:
    """Choose 'average' when coarsening, 'bilinear' when refining or matching."""
    src_res = abs(float(band.odc.geobox.resolution.x))
    return "average" if target_res > src_res else "bilinear"


def align(
    nir: xr.DataArray,
    swir: xr.DataArray,
    scl: xr.DataArray,
    target_res: float,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Put three single-date 2-D bands onto one grid at ``target_res`` metres.

    Needed for the 10 m variant (B08 at 10 m, B12 and SCL at 20 m). The 20 m
    baseline from Phase 2 already comes back on one grid.

    Rules
    -----
    * All three inputs must share a CRS (``assert_crs``), and it must be
      projected; else raise (``CRSMismatchError`` / ``ValueError``).
    * Target grid: same CRS, covering ``nir``'s bounding box, pixel size
      ``target_res``.
    * Reflectance (nir, swir): an interpolating or averaging method.
      SCL: a method that can only return values that already exist.
      Choose explicitly per band; never rely on a library default.
    * Returns ``(nir, swir, scl)`` with identical geoboxes. SCL keeps an
      integer dtype. Lazy in, lazy out.
    """
    assert_crs(swir, nir.odc.crs)
    assert_crs(scl, nir.odc.crs)

    if not nir.odc.crs.projected: raise ValueError(
        f"align needs a projected CRS (metres), got {crs!s}; "
        f"target_res={target_res} would be in degrees."
    )

    target = GeoBox.from_bbox(
        nir.odc.geobox.boundingbox, 
        crs=nir.odc.crs, 
        resolution=target_res
        )

    nir_aligned = nir.odc.reproject(target, 
                                    resampling=_reflectance_resampling(nir, target_res)
                                    )
    swir_aligned = swir.odc.reproject(target, 
                                      resampling=_reflectance_resampling(swir, target_res)
                                      )
    scl_nodata = scl.odc.nodata if scl.odc.nodata is not None else 0 #default Nan to 0 which is sentiel's nan class
    scl_aligned = scl.odc.reproject(
        target,
        resampling="nearest",
        dst_nodata=scl_nodata,
        dtype=scl.dtype, #keep dtype as int
    )

    return nir_aligned, swir_aligned, scl_aligned

# --------------------------------------------------------------------------
# YOU WRITE: 5. the whole pair
# --------------------------------------------------------------------------

def clean_pair(
    pre: xr.Dataset,
    post: xr.Dataset,
    scale_offset: Mapping[str, tuple[float, float]],
    *,
    keep: Sequence[int] = KEEP,
    nodata: int | float = NODATA,
) -> CleanPair:
    """Clean one pre-fire and one post-fire date and combine their masks.

    Parameters
    ----------
    pre, post : xr.Dataset
        Single time step each, e.g. ``ds.isel(time=0)`` from ``load_bands``.
        Raw DNs plus an ``scl`` variable.
    scale_offset : mapping band -> (scale, offset)
        From ``read_scale_offset``; one entry per reflectance band.

    Rules
    -----
    1. ``pre`` and ``post`` must be on the same grid (CRS, transform,
       shape); else ``ValueError``. Do not silently realign here.
    2. Every non-``scl`` band needs an entry in ``scale_offset``; a missing
       one -> ``ValueError`` naming it.
    3. ``valid`` = clear SCL in pre AND clear SCL in post AND every
       reflectance band has data (not nodata) in both dates.
    4. Both outputs are NaN wherever ``valid`` is False: same mask for both.
    5. Outputs drop ``scl`` and stay lazy (finish with ``assert_lazy``).
    """
    #Check for identitcal spatial grids
    if pre.odc.geobox != post.odc.geobox:
        raise ValueError(
            f"Grid mismatch between pre ({pre.odc.geobox}) and post ({post.odc.geobox})."
        )

    #Identify reflectance bands and scale_offset
    pre_bands = [b for b in pre.data_vars if b != "scl"]
    post_bands = [b for b in post.data_vars if b != "scl"]
    if set(pre_bands) != set(post_bands):
        raise ValueError(
            f"Band mismatch between pre ({pre_bands}) and post ({post_bands})."
        )

    for band in pre_bands:
        if band not in scale_offset:
            raise ValueError(
                f"Missing scale and offset entry for band '{band}' in scale_offset."
            )

    #Build Valid Mask
    valid = scl_mask(pre["scl"], keep=keep) & scl_mask(post["scl"], keep=keep)
    for band in pre_bands:
        valid = valid & (pre[band] != nodata) & (post[band] != nodata)

    #Convert to reflectance and apply the mask
    pre_clean_vars = {}
    post_clean_vars = {}
    for band in pre_bands:
        scale, offset = scale_offset[band]
        pre_refl = to_reflectance(pre[band], scale=scale, offset=offset, nodata=nodata)
        post_refl = to_reflectance(post[band], scale=scale, offset=offset, nodata=nodata)

        pre_clean_vars[band] = pre_refl.where(valid)
        post_clean_vars[band] = post_refl.where(valid)

    pre_clean = xr.Dataset(pre_clean_vars, attrs=pre.attrs)
    post_clean = xr.Dataset(post_clean_vars, attrs=post.attrs)

    #Makes sure it stays lazy
    assert_lazy(pre_clean)
    assert_lazy(post_clean)

    return CleanPair(pre=pre_clean, post=post_clean, valid=valid)


# --------------------------------------------------------------------------
# YOU WRITE: 6. how much survived?
# --------------------------------------------------------------------------

def valid_fraction(valid: xr.DataArray) -> float:
    """Fraction of grid pixels that are valid in both dates, as a float.

    This one is allowed to compute. The Phase 2 grid hugs the AOI's bounds,
    so this is the AOI's valid fraction (clipping to the exact perimeter
    polygon comes in Phase 4). Empty input -> ``ValueError``.
    """
    if valid.size == 0: #Prevents divide by 0 errors
        raise ValueError(
            "Cannot compute valid_fraction of an empty DataArray."
        )

    return float(valid.mean())
