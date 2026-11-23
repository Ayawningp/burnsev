import math
import geopandas as gpd
import numpy as np
import pyproj
from pyproj import CRS
import rasterio
from rasterio.transform import from_origin, xy
from shapely.geometry import Polygon, box
from shapely.ops import transform
from pyproj.exceptions import CRSError
from shapely.geometry.base import BaseGeometry

def utm_epsg(lon: float, lat: float) -> int:
    """returns the correct epsg code (326xx:Northern, 327xx:Southern )"""
    if not (-180 <= lon <= 180 and -80 <= lat <= 84):
        raise ValueError(
            f"Coordinates ({lon}, {lat}) outside standard UTM bounds (-180..180, -80..84)."
        )

    zone = min(60, math.floor((lon + 180) / 6) + 1) #keeps it as int and deals with edge case lon = 180
    prefix = 32600 if lat >= 0 else 32700
    return prefix + zone

class CRSMismatchError(ValueError):
    """Raised when a geospatial object's CRS is missing or does not match the expected CRS."""

_MISSING = object()

def _find_crs(obj):
    for path in (("crs",), ("rio", "crs"), ("odc", "crs")):
        # walk the attribute path with getattr(..., _MISSING)
        # if you hit _MISSING anywhere, try the next path
        # if you reach the end, return that value (even if it's None)
        curr = obj
        for attr in path:
            curr = getattr(curr, attr, _MISSING)
            if curr is _MISSING:
                break
        else:
            # Reached the end of the path; return value even if it is None
            return curr

    return obj  # nothing matched: assume obj is CRS-like itself

def assert_crs(obj, expected) -> None:
    """Asserts that a GeoDataFrame, GeoSeries, Rasterio dataset, or CRS object
    matches the expected CRS. Raises ValueError for CRS-unaware Shapely objects"""

    if expected is None:
        raise ValueError("Expected Cannot be None")

    if isinstance(obj, BaseGeometry):
        raise TypeError(
            f"Raw shapely geometry ({type(obj).__name__}) has no CRS; "
            "wrap it in a geopandas.GeoSeries or GeoDataFrame with a CRS first."
        )

    #Extract CRS attribute (Geopandas/Rasterio)
    raw_crs = _find_crs(obj)
    #Checks if from scratch Geodataframes are missing a CRS
    if raw_crs is None:
        raise CRSMismatchError(
            f"Object '{type(obj).__name__}' has no CRS assigned (crs=None)."
        )

    actual_crs = CRS.from_user_input(raw_crs)
    expected_crs = CRS.from_user_input(expected)

    #Compares CRS 
    if actual_crs != expected_crs:
        raise CRSMismatchError(
            f"CRS Mismatch! Expected '{expected_crs.to_string()}' ({expected_crs.name}), "
            f"got '{actual_crs.to_string()}' ({actual_crs.name})."
        )

def pixel_to_map(transform, row, col) -> (x, y):
    """Convert 0-based (row, col) indices to the (x, y) map coordinates of the pixel center.

    Supports full 6-parameter affine transforms (including rotation and shear)."""
    #(x,y) = (a,b;d,e) * (col, row) + (c,f)
    c_center = col + 0.5
    r_center = row + 0.5
    x = transform.c + (c_center * transform.a) + (r_center * transform.b)
    y = transform.f + (c_center * transform.d) + (r_center * transform.e)
    return (float(x), float(y))

def map_to_pixel(transform, x, y) -> (row, col):
    """Convert (x, y) map coordinates to containing 0-based (row, col) integer indices.

    Uses the inverse affine matrix so it works for both north-up and rotated rasters."""
    col_f, row_f = ~transform * (x,y)
    return (int(math.floor(row_f)), int(math.floor(col_f)))