# burnsev

A tested Python package that maps wildfire burn severity from Sentinel-2 imagery, streamed from the cloud with no full-scene downloads.

It compares a pre-fire and a post-fire scene using the differenced Normalized Burn Ratio (dNBR), classifies the result into standard severity classes, and reports burned area per class inside the fire perimeter.

## How it works

1. **Search** Earth Search (a free STAC API) for clear Sentinel-2 L2A scenes before and after the fire.
2. **Load** only the needed bands (B8A, B12, SCL) lazily as Cloud-Optimized GeoTIFFs with xarray and Dask.
3. **Clean** the data: convert to reflectance, mask clouds and shadows with the scene classification layer, and align the bands onto one UTM grid.
4. **Measure** NBR, dNBR, severity classes, and area per class clipped to the perimeter.
5. **Output** COGs with provenance metadata, plus a CSV of class areas, via a `burnsev` CLI.

Study area: the January 2025 Palisades fire, Los Angeles (UTM zone 11N, EPSG:32611).

## Status

Early development. Phase 0 (CRS utilities) is in place; the rest is planned.

| Phase | Issue |
| --- | --- |
| 0 · Coordinate reference systems | [#1](https://github.com/Ayawningp/burnsev/issues/1) |
| 1 · Scene search with STAC | [#2](https://github.com/Ayawningp/burnsev/issues/2) |
| 2 · Lazy loading from the cloud | [#3](https://github.com/Ayawningp/burnsev/issues/3) |
| 3 · Cleaning the pixels | [#4](https://github.com/Ayawningp/burnsev/issues/4) |
| 4 · Measuring burn severity | [#5](https://github.com/Ayawningp/burnsev/issues/5) |
| 5 · Output and packaging | [#6](https://github.com/Ayawningp/burnsev/issues/6) |
| 6 · Demo notebook | [#7](https://github.com/Ayawningp/burnsev/issues/7) |
| Extensions | [#8](https://github.com/Ayawningp/burnsev/issues/8) (engineering), [#9](https://github.com/Ayawningp/burnsev/issues/9) (analysis) |

The full learning roadmap, with concepts, tasks, test gates and traps for each phase, is in [docs/roadmap.md](docs/roadmap.md).

## Getting started

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Tests run offline on small synthetic fixtures; no network access or downloads needed.

## Layout

```
src/burnsev/   library code (one module per phase)
tests/         pytest suite and offline fixtures
docs/          learning roadmap
```

## Data

Sentinel-2 Level-2A surface reflectance, collection `sentinel-2-c1-l2a`, from [Element 84 Earth Search](https://github.com/Element84/earth-search). The imagery is hosted as COGs in a [public AWS bucket](https://registry.opendata.aws/sentinel-2-l2a-cogs/); no account is required.
