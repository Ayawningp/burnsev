"""Regenerate synthetic_search.json, the hand-designed STAC fixture.

    python tests/fixtures/stac/make_synthetic.py

Every item is fake (platform "S2X") but shaped like a real Earth Search
sentinel-2-c1-l2a item. Footprints are axis-aligned rectangles so the
expected answers can be checked by hand.

AOI (see test_stac.AOI_BOUNDS): lon -118.68..-118.50, lat 34.03..34.13
Fire: start 2025-01-07, containment 2025-01-31

Tiles:
  11SKT  west  lon -119.20..-118.58  covers the west part of the AOI only
  11SLT  east  lon -118.62..-117.90  covers the east part of the AOI only
  11SMT  far   lon -118.20..-117.50  does not touch the AOI
Both 11SKT and 11SLT are needed to cover the AOI.

Each row: (date, tile, cloud %, why it is in the fixture)
"""
import json
from pathlib import Path

TILES = {
    "11SKT": (-119.20, 33.50, -118.58, 34.60),
    "11SLT": (-118.62, 33.50, -117.90, 34.60),
    "11SMT": (-118.20, 33.50, -117.50, 34.60),
}

ROWS = [
    # ---- pre-fire candidates ----
    ("2024-12-20", "11SKT", 3.0, "valid pair, worst tile 4"),
    ("2024-12-20", "11SLT", 4.0, "valid pair, worst tile 4"),
    ("2024-12-28", "11SKT", 1.0, "EXPECTED PRE: valid pair, worst tile 2"),
    ("2024-12-28", "11SLT", 2.0, "EXPECTED PRE: valid pair, worst tile 2"),
    ("2025-01-02", "11SKT", 0.5, "trap: clearest tile, but its partner is cloudy"),
    ("2025-01-02", "11SLT", 60.0, "trap: fails cloud filter, so 01-02 can't cover AOI"),
    ("2025-01-05", "11SMT", 0.0, "trap: perfectly clear but misses the AOI"),
    ("2025-01-07", "11SKT", 0.0, "trap: on the fire start date, not strictly before"),
    ("2025-01-07", "11SLT", 0.0, "trap: on the fire start date, not strictly before"),
    # ---- during the fire ----
    ("2025-01-20", "11SKT", 0.0, "trap: after start, before containment"),
    ("2025-01-20", "11SLT", 0.0, "trap: after start, before containment"),
    # ---- post-fire candidates ----
    ("2025-02-03", "11SKT", 10.0, "valid pair, worst tile 12"),
    ("2025-02-03", "11SLT", 12.0, "valid pair, worst tile 12"),
    ("2025-02-06", "11SKT", 5.0, "EXPECTED POST: worst tile 5, ties 02-11, closer"),
    ("2025-02-06", "11SLT", 5.0, "EXPECTED POST: worst tile 5, ties 02-11, closer"),
    ("2025-02-11", "11SKT", 5.0, "tie with 02-06 on score, but further from fire"),
    ("2025-02-11", "11SLT", 5.0, "tie with 02-06 on score, but further from fire"),
    ("2025-02-16", "11SKT", 35.0, "trap: fails cloud filter"),
    ("2025-02-16", "11SLT", 40.0, "trap: fails cloud filter"),
]


def rect(minx, miny, maxx, maxy):
    return {
        "type": "Polygon",
        "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
    }


def make_item(day, tile, cloud, note):
    ymd = day.replace("-", "")
    item_id = f"S2X_{tile}_{ymd}_0_L2A"
    bounds = TILES[tile]
    base = f"https://example.invalid/sentinel-2-c1-l2a/{tile}/{ymd}"
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "collection": "sentinel-2-c1-l2a",
        "bbox": list(bounds),
        "geometry": rect(*bounds),
        "properties": {
            "datetime": f"{day}T18:40:12.024000Z",
            "eo:cloud_cover": cloud,
            "grid:code": f"MGRS-{tile}",
            "proj:code": "EPSG:32611",
            "platform": "sentinel-2x",
            "fixture:note": note,
        },
        "assets": {
            band: {"href": f"{base}/{band}.tif", "type": "image/tiff; application=geotiff; profile=cloud-optimized"}
            for band in ("nir08", "swir22", "scl")
        },
        "links": [],
    }


if __name__ == "__main__":
    collection = {
        "type": "FeatureCollection",
        "features": [make_item(*row) for row in ROWS],
    }
    out = Path(__file__).with_name("synthetic_search.json")
    out.write_text(json.dumps(collection, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(ROWS)} items to {out}")
