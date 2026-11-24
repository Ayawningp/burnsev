"""Save one real Earth Search response as an offline test fixture (Phase 1, task 2).

Run once, after search_items works, from the repo root:

    python scripts/save_stac_fixture.py

Writes tests/fixtures/stac/palisades_search.json. Commit it: from then on
test_real_response_obeys_gates runs offline against it.

The AOI here is an approximate bounding box of the Palisades fire, not the
official perimeter. Swap in the real perimeter (e.g. from NIFC) when you
have it, and keep the AOI in tests/test_stac.py in sync.
"""
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from burnsev.stac import search_items

AOI_BOUNDS = (-118.68, 34.03, -118.50, 34.13)  # lon/lat, approximate
PRE_WINDOW = (date(2024, 12, 1), date(2025, 1, 6))    # ends the day before ignition (2025-01-07)
POST_WINDOW = (date(2025, 2, 1), date(2025, 3, 15))   # starts the day after containment (2025-01-31)
MAX_CLOUD = 20.0

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "stac" / "palisades_search.json"

if __name__ == "__main__":
    aoi = gpd.GeoSeries([box(*AOI_BOUNDS)], crs="EPSG:4326")
    features = []
    for start, end in (PRE_WINDOW, POST_WINDOW):
        found = search_items(aoi, start, end, max_cloud=MAX_CLOUD)
        print(f"{start}..{end}: {len(found)} items")
        features.extend(found)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(features)} items to {OUT}")
