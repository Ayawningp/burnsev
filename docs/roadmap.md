# Sentinel-2 Burn Severity Pipeline: Learning Roadmap

Nov 21, 2026 · @Turtle\_Food

## Overview

You will build a tested Python package that measures wildfire burn severity from Sentinel-2 imagery, without downloading a single full scene. It is one project that trains both career directions: phases 1, 2 and 5 are data engineering, phases 3 and 4 are remote sensing, and phase 0 is the foundation under both.

**Why burn severity.** It forces every core skill at once: a vector perimeter to clip to, two bands stored at different resolutions, cloud masking, a before/after comparison, and an output other people can use. The answer is also checkable against official burn-severity maps, so you will know whether you got it right.

**Study area.** A Southern California fire from January 2025 (Palisades or Eaton). Both sit in UTM zone 11N (EPSG:32611). Caveat: much of the Eaton fire burned through neighbourhoods, where the vegetation index used here is unreliable. Start with Palisades, which burned more chaparral, and treat Eaton's urban areas as an analysis question for later.

**Data.** Sentinel-2 Level-2A (surface reflectance) from Earth Search, a free STAC API at `https://earth-search.aws.element84.com/v1`. Use the `sentinel-2-c1-l2a` collection. The imagery is Cloud-Optimized GeoTIFFs in a public AWS bucket, so no account is needed.

**How to use this doc.** Each phase has the same parts:

- **Fundamentals**: the concepts to understand before coding, intuition first.
- **Focus on**: the few things that matter most in that phase.
- **Tasks**: the work for the phase. Tick them off in the phase's GitHub issue, not here.
- **Test gates**: the phase is done only when these pass.
- **Traps**: the mistakes that produce wrong numbers without raising an error.

**The split between us (mentor mode).** Claude sets up scaffolding and writes the failing tests that define "working." You write the implementation for phases 0, 3 and 4 yourself, since those hold the core skills. Claude reviews your code against industry practice. Tests run on small offline fixtures; live runs against Earth Search happen on your laptop.

&#91;embedded content: pipeline · 6 phases on a CRS foundation\]

Data flows left to right, row by row; phase 0 sits under everything because every later phase assumes you know the CRS.

## Phase 0: Coordinate reference systems

Goal: never again trust a coordinate without knowing its CRS. This is your biggest current gap and the root of most silent geospatial bugs, so it comes first. You write all of it.

### Fundamentals

**Intuition.** A coordinate pair is meaningless on its own: (34.05, -118.25) and (385000, 3768000) can be the same spot in Los Angeles. The CRS is the decoder ring that says what the numbers mean.

- **Geographic CRS** (e.g. EPSG:4326, WGS 84): positions as angles, longitude and latitude, in degrees. Good for storing locations, bad for measuring. A degree of longitude shrinks from about 111 km at the equator to zero at the poles.
- **Projected CRS** (e.g. UTM, EPSG:32611): the curved Earth flattened onto a plane, with coordinates in metres. Every projection distorts something (area, shape, distance or direction); you pick the one that keeps what you need.
- **UTM** splits the world into 60 zones, each 6° of longitude wide. Inside its own zone, distances and shapes are nearly true. Sentinel-2 tiles are stored in the UTM zone they fall in.
- **Equal-area CRS** (e.g. EPSG:6933 or, for the US, EPSG:5070): preserves area. Use it when comparing areas across large regions or zones.
- **Datum**: the model of the Earth's shape the coordinates are measured against. WGS 84 is the common one; mixing datums shifts points by metres.
- **Axis order**: EPSG:4326 officially lists latitude first, but most software works in (lon, lat). In pyproj, always pass `always_xy=True`.

The UTM zone for a longitude λ (in degrees) is:

```latex
\text{zone} = \left\lfloor \frac{\lambda + 180}{6} \right\rfloor + 1, \qquad \text{EPSG} = \begin{cases} 32600 + \text{zone} & \text{northern hemisphere} \\ 32700 + \text{zone} & \text{southern hemisphere} \end{cases}
```

**Rasters add one more idea: the affine transform.** A raster is an array plus six numbers that map a pixel's (column, row) to map coordinates (x, y). For a north-up image, a is the pixel width, e is the negative pixel height, and (c, f) is the top-left corner:

```latex
\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} a & b \\ d & e \end{pmatrix} \begin{pmatrix} \text{col} \\ \text{row} \end{pmatrix} + \begin{pmatrix} c \\ f \end{pmatrix}
```

Two rasters line up pixel for pixel only if they share the CRS, the pixel size and the grid origin. "Same CRS" alone is not enough.

### Focus on

- Knowing which CRS every object is in at every step, and asserting it in code.
- Why area and distance must be computed in metres, never degrees.
- Reading an affine transform and converting pixel ↔ map coordinates by hand.

### Tasks

Tracked in [#1](https://github.com/Ayawningp/burnsev/issues/1).

- Write `utm_epsg(lon, lat)` returning the right EPSG code.
- Reproject a polygon from EPSG:4326 to its UTM zone with pyproj and GeoPandas.
- Compute the same polygon's area three ways: in degrees (wrong), in UTM, in an equal-area CRS. Explain the differences in a comment.
- Open any GeoTIFF with rasterio, read its `crs` and `transform`, and convert a pixel to map coordinates by hand, then check against `rasterio.transform.xy`.
- Write `assert_crs(obj, expected)` — you will reuse it in every later phase.

### Test gates

- `utm_epsg` returns 32611 for Los Angeles, and a 327xx code for a southern-hemisphere point.
- A 1 km × 1 km square built in UTM has an area of 1,000,000 m² within 0.1%.
- Pixel → map → pixel round-trips to the same (col, row).
- `assert_crs` raises on a mismatch.

### Traps

- Swapped latitude and longitude: points land in the ocean or Antarctica, with no error.
- `GeoDataFrame.set_crs` *labels* data with a CRS; `to_crs` *transforms* it. Using `set_crs` to "reproject" silently corrupts everything after it.
- GeoPandas `.area` on EPSG:4326 data returns square degrees. It only warns.

## Phase 1: Finding scenes with STAC

Goal: given a fire perimeter and dates, pick the best pre-fire and post-fire scenes automatically and reproducibly. This is a data engineering phase; Claude scaffolds it and you write the scene-selection logic.

### Fundamentals

**Intuition.** Satellite archives hold millions of images. STAC (SpatioTemporal Asset Catalog) is a shared metadata standard that lets you ask "what images cover this place, at this time, with few clouds?" without opening any image.

- **Catalog → Collection → Item → Asset.** A catalog (Earth Search) holds collections (`sentinel-2-c1-l2a`). An item is one acquisition over one tile, with a footprint, a datetime and properties like `eo:cloud_cover`. Assets are its files: one COG per band (`nir`, `swir22`, `scl`, …) plus thumbnails and metadata.
- **STAC API search** filters by `bbox` or `intersects` (a GeoJSON geometry), a `datetime` range, and property queries (e.g. cloud cover below 20%). Results are paged.
- **Search geometry is always lon/lat (EPSG:4326)**, even though the imagery itself is in UTM. Phase 0 matters here.
- **Sentinel-2 tiling.** Imagery is cut into 100 km × 100 km MGRS tiles that overlap. One date over your fire can return two or more items, and the same tile can be seen from two orbits with different footprints.
- **Revisit.** The constellation images a given spot every few days, so you will usually have several candidates per window and must choose.

### Focus on

- Separating the *network call* from the *selection logic*, so the logic is a pure function you can test offline.
- A deterministic rule for choosing scenes (e.g. lowest cloud cover over the AOI, closest to the fire date, tie-broken by item id).
- Recording which item ids you used. That is provenance, and it makes the output reproducible.

### Tasks

Tracked in [#2](https://github.com/Ayawningp/burnsev/issues/2).

- Use `pystac-client` to open Earth Search and search `sentinel-2-c1-l2a` over the fire perimeter, for a pre-fire window (weeks before the start date) and a post-fire window (after containment).
- Save one real search response as JSON. This becomes your offline test fixture.
- Write `select_scenes(items, fire_date) -> (pre, post)` as a pure function.
- Handle the case where the AOI spans more than one tile on the same date (group items by date).
- Make it fail loudly with a clear message if no scene meets the cloud threshold.

### Test gates

- Selection on the saved fixture returns the expected item ids, with no network access.
- The pre-fire scene is strictly before the fire date; the post-fire scene is after it.
- Every returned item intersects the AOI.
- An empty or all-cloudy result raises, rather than returning `None`.

### Traps

- `eo:cloud_cover` is for the *whole tile*. A tile can be 40% cloudy with your fire perfectly clear, or 5% cloudy with the cloud right over your fire. Treat it as a pre-filter; the real cloud check happens per pixel in phase 3.
- Pre and post scenes from different seasons. Vegetation greens up and dries out over the year, which shows up as false "change." Keep the windows close to the fire.

## Phase 2: Lazy loading from the cloud

Goal: load only the pixels inside the fire perimeter, for only the bands you need, without your laptop ever holding a full scene. This is the skill that makes a CPU-only laptop viable. Claude scaffolds it; you write the load function.

### Fundamentals

**Intuition.** A full Sentinel-2 band is around 10,980 × 10,980 pixels. Your fire covers a small corner of it. Cloud-native formats let you fetch just that corner over HTTP, the way a video player streams the part you are watching instead of downloading the film.

- **Cloud-Optimized GeoTIFF (COG).** An ordinary GeoTIFF arranged for the network: the image is cut into internal tiles (e.g. 512 × 512), the index of where each tile lives sits at the start of the file, and lower-resolution *overviews* are stored alongside. A reader fetches the header, then makes HTTP *range requests* for only the tiles it needs.
- **xarray** gives arrays named dimensions and coordinates: `(time, y, x)` with real map coordinates, instead of bare indices. You select by place and date, not by row number.
- **Dask** makes it lazy. Operations build a task graph; nothing is read or computed until you call `.compute()` (or write the result). Work is split into *chunks* that fit in memory.
- **odc-stac** (or stackstac) turns a list of STAC items into one lazy xarray Dataset on a grid you choose: CRS, resolution and bounds.

### Focus on

- Staying lazy until the last possible moment. One accidental `.values` or print of a whole array pulls everything into memory.
- Choosing the output grid explicitly (EPSG:32611, 20 m for the B8A baseline or 10 m for the B08 variant, bounds from the perimeter) rather than accepting defaults.
- Choosing resampling per band. If you ask odc-stac for a 10 m grid, it resamples the 20 m bands (B12, SCL) behind your back. Pick the method on purpose (see phase 3).
- Chunk size: too small and the task graph overhead dominates; too large and you run out of memory. Start near 1024 × 1024.

### Tasks

Tracked in [#3](https://github.com/Ayawningp/burnsev/issues/3).

- Load `nir08` (B8A), `swir22` (B12) and `scl` for your pre and post items with `odc.stac.load`, passing `crs`, `resolution`, `geopolygon` (or `bbox`) and `chunks`.
- Confirm the result is a Dask-backed Dataset and print its size *before* computing.
- Compute one small window and time it; then compute the whole AOI and time that.
- Build a test fixture: tiny synthetic COGs written with rasterio plus a STAC item JSON pointing at them on disk, so loading runs offline.

### Test gates

- The loaded Dataset has the expected CRS, resolution and dimensions.
- Its data variables are Dask arrays (nothing computed at load time).
- Bounds cover the AOI.
- Each band's resampling method is set explicitly in code, not left to default.

### Traps

- Loading in the wrong CRS (e.g. EPSG:4326) means every pixel later has a different size on the ground, and your area totals will be wrong.
- An AOI that crosses into a neighbouring UTM zone. Pick one target CRS and reproject everything into it.

## Phase 3: Cleaning the pixels

Goal: turn raw stored numbers into trustworthy surface reflectance, with clouds, shadows and missing data removed, and every band on one aligned grid. Most wrong answers in remote sensing are born here. You write all of it.

### Fundamentals

**Intuition.** What the file stores is not what the satellite measured. Values are packed as integers to save space, some pixels show clouds instead of ground, some are empty, and bands come at different pixel sizes. Cleaning undoes each of those before any science happens.

**1. Digital numbers → reflectance.** Reflectance is the fraction of incoming sunlight a surface reflects, from 0 to 1. Sentinel-2 L2A stores it as an integer digital number (DN). Since processing baseline 04.00 (25 January 2022), ESA adds a band-dependent offset so that slightly negative reflectances over very dark surfaces can be stored. ESA's conversion is:

```latex
\rho_{\text{BOA}} = \frac{\text{DN} + \text{BOA\_ADD\_OFFSET}}{\text{QUANTIFICATION\_VALUE}}
```

The offset is typically −1000 and the quantification value 10000, but read them from metadata rather than hard-coding: in STAC they usually appear as `scale` and `offset` under the asset's `raster:bands`. Your job is to confirm, for the collection you use, whether they have already been applied.

**2. Nodata.** Pixels outside the satellite's swath, or at tile edges, hold a fill value (0 in L2A). They are "no measurement," not "zero reflectance," and must become NaN before any maths.

**3. Cloud and shadow masking with SCL.** The Scene Classification Layer (SCL) is a 20 m band that labels each pixel. Keep only classes that show the ground:

| SCL value | Class | Keep? |
| --- | --- | --- |
| 0 | No data | No |
| 1 | Saturated or defective | No |
| 2 | Dark area / topographic shadow | No |
| 3 | Cloud shadow | No |
| 4 | Vegetation | Yes |
| 5 | Not vegetated (bare soil, burned ground) | Yes |
| 6 | Water | No (not burnable) |
| 7 | Unclassified | Your call; test both |
| 8 | Cloud, medium probability | No |
| 9 | Cloud, high probability | No |
| 10 | Thin cirrus | No |
| 11 | Snow or ice | No |

**4. Resampling.** NIR (B08) is 10 m; SWIR2 (B12) and SCL are 20 m. To combine them pixel by pixel, all must share one grid. The method depends on what the values *mean*:

- **Continuous values** (reflectance): bilinear, or average when going coarser.
- **Categorical values** (SCL classes): nearest neighbour only. Bilinear would average class 4 and class 9 into class 6.5, which means nothing.

The alternative is to work at 20 m throughout: use band 8A (20 m NIR) instead of B08, or downsample NIR to 20 m. Upsampling does not create detail that was never measured. Phase 4 starts from the 20 m baseline; the 10 m version is where this resampling work pays off.

### Focus on

- Order of operations: nodata → scale/offset → mask → resample (or resample SCL first with nearest, then mask).
- Keeping masked pixels as NaN all the way through, so they can never leak into a mean or an area count.
- Applying the *same* mask logic to the pre and post images, and then combining them: a pixel is valid only if it is clear in both.

### Tasks

Tracked in [#4](https://github.com/Ayawningp/burnsev/issues/4).

- Write `to_reflectance(da, scale, offset, nodata)`.
- Write `scl_mask(scl, keep=(4, 5))` returning a boolean mask.
- Write `align(nir, swir, scl, target_res)` with explicit resampling per band.
- Write `clean_pair(pre, post)` that applies everything and returns aligned, masked reflectance plus a combined validity mask.
- Report the fraction of the AOI that is valid in both dates. If it is low, go back to phase 1 and pick better scenes.

### Test gates

- On a synthetic array, DN 0 becomes NaN, and a known DN converts to the expected reflectance.
- Every SCL class outside `keep` is masked; every class inside is kept.
- After alignment, all bands share the same CRS, transform, shape and resolution (`assert` on `rio.transform()` equality, not just shape).
- Aligned SCL contains only integer class values that existed in the input (proves nearest-neighbour).
- Clear reflectance falls in a plausible range (roughly −0.1 to 1.2; anything far outside is a scaling bug).

### Traps

- Applying the offset twice, or not at all. Both give numbers that look reasonable and are wrong.
- Masking at 20 m and then resampling the mask with bilinear: edges turn into fractional values that pass a `> 0` check.
- Smoke. Post-fire scenes can carry smoke that SCL does not flag as cloud. Inspect a true-colour image before trusting the result.

## Phase 4: Measuring burn severity

Goal: turn clean reflectance into a burn-severity map and an area total per severity class, and know how far to trust it. This is the remote sensing core. You write all of it.

### Fundamentals

**Intuition.** Healthy leaves reflect a lot of near-infrared light (their internal cell structure scatters it) and absorb shortwave infrared (they hold water). Burned ground does the reverse: charcoal and ash are dark in NIR, and dry, exposed soil is bright in SWIR. A single number that rises with NIR and falls with SWIR therefore drops sharply where a fire burned.

**Normalized Burn Ratio.** Dividing by the sum makes it a ratio between −1 and 1, so it is less sensitive to overall brightness (sun angle, slope):

```latex
\text{NBR} = \frac{\rho_{\text{NIR}} - \rho_{\text{SWIR2}}}{\rho_{\text{NIR}} + \rho_{\text{SWIR2}}}
```

**Change, not state.** A low NBR alone could be a parking lot. What signals fire is the *drop* between before and after:

```latex
\text{dNBR} = \text{NBR}_{\text{pre}} - \text{NBR}_{\text{post}}
```

Positive dNBR means vegetation loss; values near zero mean no change; negative values can mean regrowth.

**Which bands.** Standard practice (used by EFFIS and UN-SPIDER) is Sentinel-2 band 8A (narrow NIR) with band 12 (SWIR2). Both are 20 m, so no resampling is needed. Band 08 (10 m NIR) is the alternative if you want finer pixels, at the cost of mixing resolutions. Build the 20 m B8A version first as your baseline; the 10 m version is a good experiment for phase 3's resampling work.

**Severity classes.** The thresholds from Key & Benson (2006), as used by EFFIS, with dNBR multiplied by 1000:

| Class | dNBR × 1000 |
| --- | --- |
| Unburned or regrowth | < 100 |
| Low severity | 100 to 270 |
| Moderate-low severity | 270 to 440 |
| Moderate-high severity | 440 to 660 |
| High severity | ≥ 660 |

These were calibrated on specific forests. Agencies adjust them to field observations, so treat them as a starting point, not truth.

**Area.** In a projected CRS every pixel has the same ground area, so area per class is a count:

```latex
A_k = N_k \cdot \Delta x \cdot \lvert \Delta y \rvert
```

where N\_k is the number of valid pixels in class k and Δx, Δy are the pixel sizes from the affine transform (20 m × 20 m = 400 m² per pixel at 20 m resolution).

**Clipping to the perimeter.** The fire perimeter is a vector polygon; your data is a raster. *Rasterize* the polygon onto the raster's exact grid to make a mask. Assert that both are in the same CRS first.

### Focus on

- NaN handling: a masked pixel in either date must give NaN dNBR and must not be counted in any class.
- Division by zero when NIR + SWIR2 = 0.
- Reporting how much of the perimeter was *unmeasurable* (masked) alongside the class areas. An area table without that number overstates its certainty.
- Sanity-checking against an independent source: official burn-severity products for the same fire.

### Tasks

Tracked in [#5](https://github.com/Ayawningp/burnsev/issues/5).

- Write `nbr(nir, swir2)` and `dnbr(pre, post)`.
- Write `classify(dnbr, thresholds)` with the thresholds passed in, not hard-coded.
- Write `perimeter_mask(gdf, like)` that rasterizes the perimeter onto the raster grid.
- Write `class_areas(classes, valid, transform)` returning hectares per class plus the masked fraction.
- Plot a histogram of dNBR inside vs outside the perimeter. They should look clearly different; if not, something upstream is wrong.

### Test gates

- On a 3 × 3 synthetic raster, NBR and dNBR match values you computed by hand.
- NaN in either input gives NaN output; zero denominators give NaN, not an error or infinity.
- Every threshold boundary value lands in the documented class (test 99, 100, 269, 270, …).
- Class areas plus masked area equal the perimeter's area (computed from the vector in UTM) within one pixel's worth of edge error per boundary pixel.

### Traps

- Built-up areas. Roofs, roads and burned houses do not behave like vegetation, so dNBR there is unreliable. Expect strange results inside neighbourhoods.
- Timing. A post-fire image weeks after the fire (an "initial assessment") and one a year later (an "extended assessment") measure different things. Note which you made.
- Rasterizing with the wrong grid: the mask must use the data's exact transform and shape, not just its CRS.

## Phase 5: Output and packaging

Goal: one command (`burnsev --fire palisades`) that runs the whole pipeline and writes outputs someone else can open, trust and reproduce. This is where a script becomes a data engineering product. Claude scaffolds the package; you write the output functions and the CLI.

### Fundamentals

**Intuition.** Code that only works in your notebook, on your laptop, the way you happened to run it last Tuesday, is not a pipeline. A pipeline has a defined input, a defined output, runs the same way every time, and leaves a record of what it did.

- **Writing COGs.** Your outputs (dNBR as float32, severity classes as uint8) should themselves be COGs, so the next person can stream them. Set the nodata value, keep the CRS and transform, and add overviews. With rioxarray: `da.rio.to_raster(path, driver="COG")`. Validate with `rio cogeo validate`.
- **Provenance.** Write the source STAC item ids, the thresholds, the package version and the run date into the output's metadata tags (and a small JSON sidecar). Anyone can then trace a number back to the exact scenes.
- **Package layout (src layout).** Code lives in `src/burnsev/`, tests in `tests/`, and `pyproject.toml` declares dependencies and the CLI entry point. Installing with `pip install -e .` means your tests import the package exactly as a user would.
- **CLI.** A thin layer (typer or argparse) that parses arguments and calls your library functions. Keep logic out of the CLI so it stays testable.
- **Configuration.** Fire name, dates, perimeter path and thresholds come from a config file or arguments, never edited constants.
- **Idempotence.** Running the same command twice gives the same output and does not corrupt anything. Write to a temporary file, then rename.

### Focus on

- The boundary between library and CLI.
- Making every run reproducible from its inputs plus the provenance record.
- Logging what happened (scenes chosen, valid fraction, class areas) instead of printing.

### Tasks

Tracked in [#6](https://github.com/Ayawningp/burnsev/issues/6).

- Write `write_cog(da, path, nodata, tags)`.
- Write the provenance sidecar (JSON with item ids, thresholds, version, timestamp).
- Wire phases 1–4 into `run(config) -> Result`.
- Add the `burnsev` CLI entry point in `pyproject.toml`.
- Add a CSV of class areas as a third output.

### Test gates

- Written outputs pass COG validation.
- Reading an output back gives the same CRS, transform, nodata and values that were written.
- The CLI runs end to end on the offline fixtures (typer's `CliRunner` or a subprocess call) and exits 0.
- Two runs on the same inputs produce identical data.
- A bad argument (unknown fire, perimeter in the wrong CRS) exits non-zero with a clear message.

### Traps

- Writing classes as float with NaN nodata: tools that expect integer classes will misread it. Use uint8 with an explicit nodata value such as 255.
- Forgetting overviews: the file is a valid GeoTIFF but slow to view remotely.

## Phase 6: Demo notebook

Goal: a short notebook that shows a reader what the package does and what it found, in under five minutes of reading. It imports the package; it contains no pipeline logic of its own. You write it.

### Fundamentals

**Intuition.** The notebook is the shop window, not the workshop. If a bug fix ever has to happen in the notebook, the logic was in the wrong place.

- **Imports, not copies.** `from burnsev import run, class_areas`. The package is the single source of truth.
- **Telling the result.** A reader needs: where and when, which scenes, the severity map over a basemap, the area table including the unmeasured fraction, and the caveats (urban areas, thresholds, timing).
- **Maps.** matplotlib with the correct CRS-aware extent is enough for static maps; leafmap or folium give an interactive basemap. Use a colour scale with ordered, distinguishable classes.

### Focus on

- Clear story over code volume.
- Honest caveats next to the numbers.
- Restart-and-run-all working cleanly.

### Tasks

Tracked in [#7](https://github.com/Ayawningp/burnsev/issues/7).

- Show the pre and post true-colour images side by side (also your visual smoke check).
- Show the dNBR histogram and the classified map clipped to the perimeter.
- Show the class-area table with the masked fraction.
- Compare against an official burn-severity product for the same fire, even if only visually.

### Test gates

- The notebook runs top to bottom on a fresh kernel (e.g. `jupyter nbconvert --execute`).
- It contains no function definitions that duplicate package code.

## Branch points: choosing a direction

After phase 6 you have one working pipeline. Pick one extension from each column, do one, and notice which you wanted to keep going on. That is better evidence for a career direction than any article.

|  | Engineering branch | Analysis branch |
| --- | --- | --- |
| The question becomes | How do I run this for every fire, reliably and cheaply? | Is this map actually right, and what does it mean? |
| Extension 1 | Run for a list of California fires; store perimeters and results as GeoParquet | Validate against official USGS MTBS severity maps; build a confusion matrix |
| Extension 2 | Schedule runs (cron, GitHub Actions or Prefect) with retries and a run log | Add RdNBR (relative dNBR, which MTBS also computes for low-biomass sites) and compare |
| Extension 3 | Publish outputs as a STAC catalog of your own | Train a classifier on the bands with spatially blocked CV; compare with threshold classes |
| New skills | Orchestration, storage formats, catalogs, cost and scale | Radiometry, validation statistics, spatial ML |
| Typical roles | Geospatial data engineer, platform engineer | Remote sensing scientist, geospatial data scientist |

Signals to watch in yourself: if the most satisfying moment was a clean, fast, reproducible run, lean engineering. If it was explaining *why* one patch of hillside scored high, lean analysis. Many jobs mix both, and the overlap (phases 0, 2 and 3) is valuable either way.

## Repo layout, tools and progress

### Repo layout

```
burnsev/
├── pyproject.toml
├── src/burnsev/
│   ├── crs.py        # phase 0: utm_epsg, assert_crs
│   ├── search.py     # phase 1: STAC query, select_scenes
│   ├── load.py       # phase 2: lazy load
│   ├── clean.py      # phase 3: reflectance, masks, alignment
│   ├── severity.py   # phase 4: nbr, dnbr, classify, areas
│   ├── io.py         # phase 5: COG writing, provenance
│   └── cli.py        # phase 5: burnsev command
├── tests/
│   ├── fixtures/     # saved STAC JSON, tiny synthetic COGs
│   └── test_*.py     # one file per module
└── notebooks/demo.ipynb   # phase 6
```

### Tools by phase

| Phase | Libraries |
| --- | --- |
| 0 | pyproj, shapely, geopandas, rasterio |
| 1 | pystac-client |
| 2 | odc-stac (or stackstac), xarray, dask, rioxarray |
| 3–4 | numpy, xarray, rasterio.features (rasterize) |
| 5 | rioxarray, rio-cogeo, typer, pytest |
| 6 | matplotlib, leafmap or folium, jupyter |

### Progress

Tracked in GitHub issues:

- [#1](https://github.com/Ayawningp/burnsev/issues/1) Phase 0: CRS fundamentals
- [#2](https://github.com/Ayawningp/burnsev/issues/2) Phase 1: STAC search and scene selection
- [#3](https://github.com/Ayawningp/burnsev/issues/3) Phase 2: Lazy loading
- [#4](https://github.com/Ayawningp/burnsev/issues/4) Phase 3: Cleaning
- [#5](https://github.com/Ayawningp/burnsev/issues/5) Phase 4: Burn severity
- [#6](https://github.com/Ayawningp/burnsev/issues/6) Phase 5: Output and packaging
- [#7](https://github.com/Ayawningp/burnsev/issues/7) Phase 6: Demo notebook
- [#8](https://github.com/Ayawningp/burnsev/issues/8) One engineering extension
- [#9](https://github.com/Ayawningp/burnsev/issues/9) One analysis extension

### Sources

- [EFFIS: Fire severity](https://forest-fire.emergency.copernicus.eu/about-effis/technical-background/fire-severity) — dNBR classes and Sentinel-2 bands 8A and 12
- [ESA: Level-2A algorithms and products](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a-algorithms-products) — BOA\_ADD\_OFFSET from baseline 04.00
- [Element 84 Earth Search](https://github.com/Element84/earth-search) and [Sentinel-2 COGs on AWS](https://registry.opendata.aws/sentinel-2-l2a-cogs/) — collections and access
- [MTBS mapping methods](https://mtbs.gov/mapping-methods) — official severity maps and RdNBR
- [UN-SPIDER burn severity practice](https://github.com/UN-SPIDER/burn-severity) — a worked reference implementation
