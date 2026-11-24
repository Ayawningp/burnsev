"""Phase 1: finding Sentinel-2 scenes with STAC.

Two jobs, deliberately kept apart:

* ``search_items`` makes the network call to Earth Search and returns raw
  STAC item dicts. It is thin and not unit-tested beyond its input checks.
* ``select_scenes`` is a pure function: items in, (pre, post) out. No network,
  no clock, no randomness, so it is tested offline against saved JSON.

Items are plain STAC item dicts (GeoJSON Features), exactly what the API
returns. If you have ``pystac.Item`` objects, pass ``item.to_dict()``.

STAC geometries are always lon/lat (EPSG:4326, per the GeoJSON spec), so the
AOI passed to both functions must be in EPSG:4326 too, even though the imagery
itself is in UTM.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Sequence
from typing import Any
from collections import defaultdict

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
#Use union_all for geopandas object and unary_union for python collections

from burnsev.crs import assert_crs  # noqa: F401  (you will need it below)

from pystac_client import Client

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
STAC_CRS = "EPSG:4326"


class NoSceneError(RuntimeError):
    """No date group meets the cloud, AOI-intersection and coverage rules."""


# --------------------------------------------------------------------------
# Helpers (done for you)
# --------------------------------------------------------------------------

def load_items(path: str | Path) -> list[dict]:
    """Read a saved STAC search response (a FeatureCollection) into item dicts."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection")
    return list(data["features"])


def acquisition_date(item: dict) -> date:
    """UTC calendar date of the acquisition, from ``properties.datetime``."""
    stamp = item["properties"]["datetime"].replace("Z", "+00:00")
    return datetime.fromisoformat(stamp).date()


def footprint(item: dict) -> BaseGeometry:
    """Item footprint as a shapely geometry, in lon/lat (EPSG:4326)."""
    return shape(item["geometry"])


# --------------------------------------------------------------------------
# YOU WRITE: Phase 1 task 1 (network call)
# --------------------------------------------------------------------------

def search_items(
    aoi,
    start: date,
    end: date,
    *,
    max_cloud: float = 20.0,
    url: str = EARTH_SEARCH_URL,
    collection: str = COLLECTION,
) -> list[dict]:
    """Search Earth Search for items over ``aoi`` between ``start`` and ``end``.

    Parameters
    ----------
    aoi : geopandas.GeoSeries or GeoDataFrame
        Fire perimeter. Must be in EPSG:4326; check this with ``assert_crs``
        *before* any network call (the tests run with the network blocked).
    start, end : date
        Inclusive date window.
    max_cloud : float
        Pre-filter on tile-level ``eo:cloud_cover`` (percent).

    Returns
    -------
    list of STAC item dicts (the ``features`` of the response).

    Hints: ``pystac_client.Client.open(url)``, ``client.search(collections=...,
    intersects=..., datetime="YYYY-MM-DD/YYYY-MM-DD", query={...})``, and
    ``search.item_collection_as_dict()``. ``intersects`` wants a GeoJSON
    geometry dict: look at ``aoi.union_all().__geo_interface__``.
    """

    #Validate CRS
    assert_crs(aoi, "EPSG:4326")

    #prepare query parameters
    intersects_geom = aoi.union_all().__geo_interface__
    datetime_range = f"{start.isoformat()}/{end.isoformat()}"
    cloud_query = {"eo:cloud_cover": {"lte": max_cloud}}

    #Execute STAC API search
    client = Client.open(url)
    search = client.search(
        collections=[collection],
        intersects=intersects_geom,
        datetime=datetime_range,
        query=cloud_query,
    )

    #Return the list of feature dicts
    feature_collection = search.item_collection_as_dict()
    return feature_collection["features"]



# --------------------------------------------------------------------------
# YOU WRITE: Phase 1 core task (pure selection logic)
# --------------------------------------------------------------------------

def select_scenes(
    items: Sequence[dict],
    aoi,
    fire_start: date,
    fire_end: date | None = None,
    *,
    max_cloud: float = 20.0,
) -> tuple[list[dict], list[dict]]:
    """Pick the pre-fire and post-fire scenes, deterministically.

    Parameters
    ----------
    items : sequence of STAC item dicts
        Mixed pre and post candidates, in any order. Must not be mutated.
    aoi : geopandas.GeoSeries or GeoDataFrame in EPSG:4326
        Assert its CRS before using it in any overlay.
    fire_start : date
        Ignition date. Pre-fire scenes must be strictly before it.
    fire_end : date, optional
        Containment date. Post-fire scenes must be strictly after it.
        Defaults to ``fire_start``.
    max_cloud : float
        Items with ``eo:cloud_cover`` above this are dropped (percent).

    Rules (the tests check each one)
    --------------------------------
    1. Drop items whose footprint does not intersect the AOI.
    2. Drop items with ``eo:cloud_cover > max_cloud``.
    3. Group the remaining items by ``acquisition_date``. A date group is
       usable only if the union of its footprints covers the whole AOI
       (the fire can straddle two MGRS tiles).
    4. Score each usable group by its *worst* tile: the max cloud cover.
    5. Pre: among groups strictly before ``fire_start``, lowest score wins;
       ties go to the date closest to the fire (the latest).
       Post: among groups strictly after ``fire_end``, lowest score wins;
       ties go to the date closest to the fire (the earliest).
    6. Return ``(pre_items, post_items)``, each a list sorted by item ``id``.
    7. If either side has no usable group, raise ``NoSceneError`` with a
       message naming the side ("pre-fire" / "post-fire") and ``max_cloud``.
       Never return ``None`` or an empty list.
    """

    #Validate CRS
    assert_crs(aoi, "EPSG:4326")

    #If fire_end is NONE defaults to fire_start
    if fire_end is None:
        fire_end = fire_start

    aoi_geom = aoi.union_all()

    #Group valid candidate records by acquisition date
    by_date: dict[date, list[tuple[dict[str, Any], Any, float]]] = defaultdict(list)

    for item in items:
        props = item["properties"]
        cloud = float(props["eo:cloud_cover"])

        #Drop items with eo:cloud_cover > max_cloud
        if cloud > max_cloud:
            continue

        geom = shape(item["geometry"])

        #Drop items whose footprint does not intersect the AOI.
        if not geom.intersects(aoi_geom):
            continue

        acq_date = acquisition_date(item)
        by_date[acq_date].append((item, geom, cloud))

    #Filter the date groupswholes footprints cover the whole AOI
    #Score by its worst tile: max cloud cover
    usable_groups: dict[date, tuple[float, list[dict[str, Any]]]] = {}

    for acq_date, records in by_date.items():
        combined_footprint = unary_union([rec[1] for rec in records])
        if combined_footprint.covers(aoi_geom):
            worst_cloud = max(rec[2] for rec in records)
            group_items = [rec[0] for rec in records]
            usable_groups[acq_date] = (worst_cloud, group_items)

    #Spilt into pre and post fire groups
    pre_dates = [d for d in usable_groups if d < fire_start]
    post_dates = [d for d in usable_groups if d > fire_end]

    #Raise NoSceneError if either side is empty while naming max_cloud
    if not pre_dates:
        raise NoSceneError(
            f"No usable pre-fire scene group found covering the AOI with max_cloud={max_cloud}%."
        )
    if not post_dates:
        raise NoSceneError(
            f"No usable post-fire scene group found covering the AOI with max_cloud={max_cloud}%."
        )

    #Tiebreaking rule
    #Pre_fire(fire_start): lowest score wins, ties go to date closet to the fire (latest)
    best_pre_date = min(
        pre_dates,
        key= lambda d: (usable_groups[d][0], -d.toordinal()), #maximize d.toordinal()
    )

    #Post_fire(fire_end): lowest score wins, ties go to date closest to the fire (earliest)
    best_post_date = min(
        post_dates,
        key= lambda d: (usable_groups[d][0], d.toordinal()) #minimize d.toordinal()
    )

    #Return a list sorted by id
    pre_items = sorted(usable_groups[best_pre_date][1], key=lambda it: it["id"])
    post_items = sorted(usable_groups[best_post_date][1], key=lambda it: it["id"])

    return pre_items, post_items
