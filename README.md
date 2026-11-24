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

Download `ppp_2020_1km_Aggregated.tif` to `data/raw/`. Poppyn aggregates, redistributes and rounds the WorldPop and CIESIN 2020 population estimates to produce lossless PNG tiles and a JSON manifest. [Dataset](https://hub.worldpop.org/geodata/summary?id=24777) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Generator design](docs/v3-plan.md)

Tifffile can log a `GDAL_NODATA` warning when opening this raster. Poppyn reads the tag directly at the raster’s precision so that it masks the correct pixels.

On macOS, GPU builds require Xcode’s Metal Toolchain. Install it with `xcodebuild -downloadComponent metalToolchain`. On Linux, follow Modular’s driver requirements for your GPU. [Apple’s installation guide](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components) · [Modular’s GPU requirements](https://max.modular.com/gpu/hardware-compatibility/)

Build the Mojo executable and check it against the CPU reference.

```sh
pixi run --locked verify-gpu
```

```sh
pixi run --locked generate --tile-width 480 --max-width 7680 --pop-target 62500 --annotated
```

By default, this reads `data/raw/ppp_2020_1km_Aggregated.tif` and runs `.pixi/poppyn-redistribute`. Run `pixi run --locked build-gpu` if the executable is missing. Supply a GeoTIFF as the first positional argument or use `--gpu PATH` to override these paths.

`--tile-width` sets the width of the smallest map and the edge length of each tile. Poppyn doubles the map width until it reaches `--max-width`. If the next doubling would go past the maximum, it uses the exact maximum for the last map. Each map follows the source window's aspect ratio. `--pop-target` sets how many people a bright pixel represents on the largest map. Coarser maps represent proportionally more people per bright pixel based on the number of grid cells, rounded to the nearest person.

The example writes `data/output/v3/world_480_7680_62500/tiles/` with subdirectories `480/`, `960/`, `1920/`, `3840/` and `7680/`, and writes the audit to `data/output/v3/world_480_7680_62500/report.json`. The CLI prints the destination, prepares all maps and confirms the path once it publishes the catalogue.

Pass a catalogue directory after the GeoTIFF argument to choose another destination, and use `--report` to save the audit at a chosen path. For a quick CPU run, use `--cpu --tile-width 480 --max-width 480 --pop-target 16000000`. The Romania correction is applied where the run overlaps the affected cells of the original WorldPop grid. Pass `--no-romania-patch` for the original estimates.

### Named slices

Use `--slice` with `europe`, `us`, `india`, `fareast`, `britain`, `useastcoast`, `japan`, `london` or `eastasia`. The original v2 windows and the East Asia window are defined in source raster coordinates. Poppyn selects the window before redistribution. Supply the desired map widths for that window. [Slice definitions](poppyn/statics.py)

```sh
pixi run --locked generate --slice eastasia --tile-width 727 --max-width 727 --pop-target 62500
```

This produces a 727 × 406 grid under `data/output/v3/eastasia_727_727_62500/tiles/727/`. Each bright pixel represents 62,500 people.

For a custom slice, provide both a distinct `--slice` name and `--slice-idx TOP BOTTOM LEFT RIGHT`. The indices are zero-based in the source TIFF. The bottom and right limits are exclusive. Names are lowercased and spaces become hyphens in the run directory.

```sh
pixi run --locked generate --slice "Tokyo Region" --slice-idx 4500 6500 37100 39100 \
  --tile-width 356 --max-width 356 --pop-target 62500 --annotated
```

### Full images, raw heatmaps and reruns

Each width directory includes a full-resolution `map.png` beside its numbered tile columns. `--annotated` adds `map-annotated.png` with a caption below the map, including the population scale, pixel area at the map centre and credits. The Tokyo Region command writes both PNGs under `data/output/v3/tokyo-region_356_356_62500/tiles/356/`.

Run the same command with `--raw` to export a heatmap of people per pixel before redistribution. It uses the same source window and grid, with a logarithmic colour scale whose maximum is recorded in the manifest. The raw export runs without a GPU executable. Its own run directory uses a `raw_` prefix and contains `tiles/` and `report.json`. For example, the world command with `--raw` writes `data/output/v3/raw_world_480_7680_62500/`. `--annotated` also works for this export.

Poppyn caches aggregated grids and solved binary maps under `data/cache/v3/`. The cache checks the source contents, window, levels, correction and solver settings. Annotation reuses the solved map, and a raw export can reuse the aggregated grids. Rerunning refreshes the generated catalogue with the requested outputs. Use `--force-rerun` to recompute and refresh the cache as well.

`pixi run --locked generate --help` lists all options for the `run_all.py` entry point.
