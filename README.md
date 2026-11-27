# Poppyn
### Binary Population Data Generation and Plotting

Heatmaps of population have a lot of data, and look awesome. 
But they are hard to parse.
For example, no matter how good your eye is, and how good the legend in the heatmap is,
you can't intuitively tell that two orange pixels are equal to one red pixel.

Poppyn processes population data into a binary form.
Either the area has population >N, or it doesn't.

But it isn't quite that simple. 
This would only show what areas of the world are densely populated.
Poppyn 'flattens' the data, so densely populated areas will spill over,
and sparsely populated areas will group together.
This loses some local geographical accuracy,
but helps massively for simple visual comparison of population.

See full justification on [this blog-post](https://medium.com/@naspli/making-spatial-population-data-easier-to-comprehend-053e6035c000)

## Set up Poppyn

Install Pixi with `brew install pixi` on macOS, then run these commands from the repository root. Pixi installs Python, the data-processing libraries, Mojo and MAX together in `.pixi/envs/default` using the versions recorded in `pixi.lock`. The manifest covers Apple silicon and Linux on x86-64 and ARM64. [Pixi installation guide](https://pixi.prefix.dev/latest/installation/)

```sh
pixi install --locked
pixi run --locked test
```

## Generate maps

Download `global_pop_2025_CN_1km_R2025A_v1.tif` to `data/raw/`. Poppyn aggregates, redistributes and rounds the 2025 population estimates to produce a PNG map. [Download GeoTIFF](https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/0_Mosaicked/v1/1km/constrained/global_pop_2025_CN_1km_R2025A_v1.tif)

Cite Bondarenko et al. (2025), WorldPop, University of Southampton, *Global2 global population counts at 30 arc seconds, R2025A v1*, 2025 estimate. Annotated maps include the dataset DOI `10.5258/SOTON/WP00845`. [Dataset citation](https://hub.worldpop.org/doi/10.5258/SOTON/WP00845) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

To use the legacy 2020 source, pass `data/raw/ppp_2020_1km_Aggregated.tif` explicitly. Tifffile can log a `GDAL_NODATA` warning for that raster. Poppyn reads the tag directly at the raster’s precision. [Legacy dataset](https://hub.worldpop.org/geodata/summary?id=24777)

On macOS, GPU builds require Xcode’s Metal Toolchain. Install it with `xcodebuild -downloadComponent metalToolchain`. On Linux, follow Modular’s driver requirements for your GPU. [Apple’s installation guide](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components) · [Modular’s GPU requirements](https://max.modular.com/gpu/hardware-compatibility/)

Build the Mojo executable and check it against the CPU reference.

```sh
pixi run --locked verify-gpu
```

```sh
pixi run --locked generate --width 7680 --pop-target 62500 --annotated
```

By default, this reads the Global2 GeoTIFF from `data/raw/` and runs `.pixi/poppyn-redistribute`. Run `pixi run --locked build-gpu` if the executable is missing. Supply a GeoTIFF as the first positional argument or use `--gpu PATH` to override these paths.

`--width` sets the map width in pixels. The height follows the source window's aspect ratio. `--pop-target` sets how many people a bright pixel represents. The example writes `data/output/v3/world_7680_62500/poppyn-map.png`. Add `--annotated` to draw a translucent title, population scale, image dimensions, pixel area and credits directly on that PNG.

Before projection and downsampling, Poppyn masks open lakes using the bundled Natural Earth 1:10 million lake outlines. It keeps the original WorldPop estimates within two source cells of a mapped shoreline, including around islands. A cell is masked when its centre and all the centres in its surrounding 5 × 5 neighbourhood are in mapped water. The population estimates in those cells are removed from the total, and `--report` records the reduction as `lakePopulationRemovedPeople`. This applies to binary maps and raw heatmaps. [Natural Earth v5.1.2 source](https://github.com/nvkelso/natural-earth-vector/blob/v5.1.2/geojson/ne_10m_lakes.geojson) · [Public domain terms](https://www.naturalearthdata.com/about/terms-of-use/)

Downsampling shares each remaining WorldPop cell's population among the output cells it overlaps, in proportion to the overlap in the raster grid. This assumes an even spread within each source cell and preserves the corrected total. An output cell can receive population during redistribution if it overlaps a valid, unmasked source cell, including one with zero estimated population.

The map background also uses the Natural Earth 1:10 million land and lake outlines. Inland cells with missing population data use the same colour as empty land. The outlines are drawn on each output grid for both binary maps and heatmaps. Annotated maps credit Natural Earth in the footer. [Land outlines](https://github.com/nvkelso/natural-earth-vector/blob/v5.1.2/geojson/ne_10m_land.geojson) · [Lake outlines](https://github.com/nvkelso/natural-earth-vector/blob/v5.1.2/geojson/ne_10m_lakes.geojson) · [Public domain terms](https://www.naturalearthdata.com/about/terms-of-use/)

### Optional projection

Add `--projection equal-area` to use the cylindrical equal-area projection EPSG:6933. Poppyn selects the source window and applies the lake mask, then aggregates directly onto the projected grid before redistribution. The map height follows the projected aspect ratio, and each pixel covers the same ground area. The projection compresses north–south distances towards the poles. [Projection reference](https://nsidc.org/data/user-resources/help-center/guide-ease-grids)

```sh
pixi run --locked generate --width 7680 --pop-target 62500 --projection equal-area --annotated
```

This writes `data/output/v3/world_7680_62500_equal-area/poppyn-map.png`. The projection works with named and custom slices, raw heatmaps and tiles. Projected tile manifests and reports record the `crs` and `projectedBounds` in metres. The `bounds` field in tile manifests still gives the source window in longitude and latitude. `--projection geographic` is the default.

Use `--projection equal-earth` for the curved Equal Earth outline. The world map extends to both poles, and the area beyond its outline has a separate background shade. Redistribution stops at the curved world seam. [Equal Earth reference](https://proj.org/en/stable/operations/projections/eqearth.html)

```sh
pixi run --locked generate --width 7680 --pop-target 62500 --projection equal-earth --annotated
```

This writes `data/output/v3/world_7680_62500_equal-earth/poppyn-map.png`. For a named or custom slice, Equal Earth uses the slice’s midpoint longitude as its central meridian. The output uses the smallest projected rectangle enclosing the slice. Population in the extra area, including across the antimeridian where applicable, enters the run. The report and tile manifest record the central meridian and projected extent. The report's `sourceWindow` gives the slice's source pixel indices before projection.

Equal Earth divides each source row into two latitude bands and uses their mean projected width to approximate the curved cell edges. Each band retains its ground area, and the aggregate is checked against the population included in the projected frame at every resolution.

### Optional tiling

Add `--tile-width 675` to write `tiles_675/` beside the full map. Split `0` starts at a map width of 675 pixels or the requested width if smaller. Each subsequent split doubles that width, with the last split using the exact requested width. Coarser maps represent proportionally more people per bright pixel based on the number of grid cells. Each split contains only image tiles, at `{split}/{column}/{row}.png`, and edge tiles can be smaller than 675 pixels. The manifest in `tiles_675/` records the geographic bounds, map sizes and population scale for each split. The caption is drawn on the full image.

Pass an output directory after the GeoTIFF argument to choose another destination, and use `--report PATH` to save the audit and source filename separately. For a quick CPU run, use `--cpu --width 480 --pop-target 16000000`.

Redistribution continues while excess population falls measurably. It stops successfully at the overflow tolerance and reports a stall if the excess falls by less than 0.0001 bright pixels' worth of population over at least 4,096 rounds. Use `--max-passes` to set a separate hard limit for a run.

### Named slices

Use `--slice` with a preset such as `europe`, `eastasia`, `south-america` or `nile-delta`. Run `pixi run --locked generate --help` for the full list. Presets use WGS 84 bounds in `(west, south, east, north)` order. Poppyn snaps those bounds outwards to source cell edges within the input raster. Supply the desired map width for the region. [Slice definitions](poppyn/statics.py)

```sh
pixi run --locked generate --slice eastasia --width 727 --pop-target 62500
```

This produces a 727 × 406 map at `data/output/v3/eastasia_727_62500/poppyn-map.png`. Each bright pixel represents 62,500 people.

For a custom slice, provide both a distinct `--slice` name and `--slice-idx TOP BOTTOM LEFT RIGHT`. The indices are zero-based in the source TIFF. The bottom and right limits are exclusive. Names are lowercased and spaces become hyphens in the run directory.

```sh
pixi run --locked generate --slice "Tokyo Region" --slice-idx 4500 6500 37100 39100 \
  --width 356 --pop-target 62500 --annotated
```

### Raw heatmaps and reruns

Run the same command with `--raw` to export a heatmap of people per pixel after applying the lake mask and before redistribution. It uses a logarithmic colour scale and runs without a GPU executable. The world command with `--raw` writes `data/output/v3/raw_world_7680_62500/poppyn-map.png`. Annotation and tiling options also work with this export.

Poppyn caches aggregated grids and solved binary maps under `data/cache/v3/`. The cache checks the source contents, lake outlines, window, resolutions and solver settings. Annotation reuses the solved map, and a raw export can reuse the aggregated grids. Rerunning replaces the generated map and tile set with the files requested by the latest command. Use `--force-rerun` to recompute and refresh the cache as well.

`pixi run --locked generate --help` lists all options for the `run_all.py` entry point.
