"""Generate a Poppyn population map from a WorldPop GeoTIFF."""

import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
from subprocess import CalledProcessError
from time import monotonic

import numpy as np

from .cache import ArrayCache, cache_key, file_digest
from .output import catalogue_stage, publish_catalogue, write_json
from .population import Population, levels_for_source, read_population
from .redistribution import child_blocks, round_marks, settle
from .render import MAP_FILENAME, annotate, binary_image, heatmap_image, write_map, write_tiles
from .statics import WORLD_SLICES


DEFAULT_SOURCE = Path("data/raw/ppp_2020_1km_Aggregated.tif")
DEFAULT_GPU = Path(".pixi/poppyn-redistribute")
SLICE_TITLES = {"us": "US", "useastcoast": "US East Coast", "fareast": "Far East", "eastasia": "East Asia"}


def area_title(name):
    return SLICE_TITLES.get(name, name.replace("-", " ").replace("_", " ").title())


def pinholes(land, marks):
    interior = land[1:-1, 1:-1] & ~marks[1:-1, 1:-1]
    height, width = interior.shape
    for y, x in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)):
        interior &= marks[y:y + height, x:x + width]
    return int(interior.sum())


def local_errors(initial, settled, marks):
    """Compare count error before and after redistribution at several scales."""
    reports = []
    size = 1
    marks = marks.astype(np.int32)
    while initial.shape != (1, 1) and size < 128:
        size *= 2
        initial, settled, marks = (child_blocks(grid).sum(axis=(2, 3))
                                   for grid in (initial, settled, marks))
        if size in (2, 8, 32, 128):
            reports.append({
                "blockWidth": size,
                "meanAbsoluteNetErrorMarks": float(np.abs(marks - initial).mean()),
                "maxAbsoluteNetErrorMarks": float(np.abs(marks - initial).max()),
                "maxAbsoluteRoundingErrorMarks": float(np.abs(marks - settled).max()),
            })
    return reports


def population_for_run(source, levels, cache, key, **reader_options):
    fields = [f"{kind}_{level.width}" for level in levels for kind in ("counts", "land")]
    cached = cache.load(f"population-{key}", "metadata", *fields)
    if cached is not None:
        metadata, *arrays = cached
        bounds, total, correction = json.loads(metadata.item())
        print("Reusing population estimates from an earlier run.", flush=True)
        return Population(tuple(bounds), total, correction, list(zip(arrays[::2], arrays[1::2])))

    print("Reading WorldPop estimates and grouping them into map pixels.", flush=True)
    population = read_population(source, levels, **reader_options)
    cache.save(f"population-{key}",
               metadata=np.array(json.dumps([population.bounds, population.population, population.correction])),
               **dict(zip(fields, (array for grid in population.grids for array in grid))))
    return population


def solve_level(level, counts, land, wrap_x, cache, key, solver_options):
    cached = cache.load(f"binary-{key}", "marks", "audit")
    if cached is not None:
        marks, audit = cached
        print(f"Reusing the {level.width:,}-pixel map from an earlier run.", flush=True)
        return marks, {**json.loads(audit.item()), "redistributionCacheHit": True}

    def progress(passes, overflow, drift):
        print(f"  After {passes:,} rounds, dense pixels still hold about {overflow * target:,.0f} "
              f"people above their capacity. The total differs by {drift * target:+,.1f} people "
              "due to numerical rounding.", flush=True)

    target = level.people_per_mark
    print(f"Spreading population on the {level.width:,} by {level.height:,} map. "
          f"One bright pixel represents {target:,} people.", flush=True)
    solved = settle(counts, land, target, wrap_x=wrap_x, progress=progress, **solver_options)
    total = float(counts.sum(dtype=np.float64))
    marks = round_marks(solved.grid, land, math.floor(total / target + 0.5))
    number_of_marks = int(marks.sum())
    initial_mass = counts / target
    settled_mass = np.where(land, solved.grid, 0).astype(np.float64)
    audit = {
        "aggregatedPopulation": total,
        "passes": solved.passes,
        "redistributionSeconds": solved.seconds,
        "redistributionCacheHit": False,
        "overflowMarks": solved.overflow,
        "massDriftMarks": solved.mass_drift,
        "floatConversionDriftMarks": solved.conversion_drift,
        "marks": number_of_marks,
        "peopleRepresented": number_of_marks * target,
        "populationRoundingResidualPeople": total - number_of_marks * target,
        "netRedistributionPeople": float(np.abs(settled_mass - initial_mass).sum() * target / 2),
        "eligiblePinholes": pinholes(land, marks),
        "localErrors": local_errors(initial_mass, np.minimum(settled_mass, 1), marks),
    }
    cache.save(f"binary-{key}", marks=marks, audit=np.array(json.dumps(audit)))
    print(f"Placed {number_of_marks:,} bright pixels representing about "
          f"{number_of_marks * target:,} people. Spreading took {solved.seconds:.1f} seconds.", flush=True)
    return marks, audit


def generate(source, output, levels, *, tile_width=None, romania_patch=True, slice_name="world", slice_window=None,
             annotated=False, raw=False, cache_dir=None, force_rerun=False, report_path=None, **solver_options):
    if slice_window is not None and (not slice_name or slice_name == "world"):
        raise ValueError("custom slices require a name")
    output = output.parent.resolve() / output.name
    if report_path is not None:
        report_path = report_path.parent.resolve() / report_path.name
        if report_path == output:
            raise ValueError("report path must differ from the catalogue directory")
    with catalogue_stage(output, report_path, raw=raw) as staged:
        report = render_catalogue(staged, source, levels, tile_width, romania_patch, slice_name, slice_window,
                                  annotated, raw, ArrayCache(cache_dir, force_rerun), solver_options)
        publish_catalogue(staged, output, report, report_path)
        return report


def render_catalogue(staged, source, levels, tile_width, romania_patch, slice_name, slice_window, annotated, raw,
                     cache, solver_options):
    start = monotonic()
    window = slice_window if slice_window is not None else WORLD_SLICES.get(slice_name)
    population_key = binary_key = ""
    if cache.directory is not None:
        code = Path(__file__).parent
        population_key = cache_key(
            file_digest(source), window, slice_window is None and slice_name != "world", romania_patch,
            [level.metadata() for level in levels], np.__version__, version("tifffile"), version("imagecodecs"),
            file_digest(code / "population.py"), file_digest(code / "statics.py"), file_digest(Path(__file__)),
        )
        if not raw:
            options = solver_options.copy()
            gpu = options.pop("gpu", None)
            binary_key = cache_key(population_key, options, file_digest(Path(gpu)) if gpu else "numpy",
                                   file_digest(code / "redistribution.py"))
    population = population_for_run(source, levels, cache, population_key, romania_patch=romania_patch,
                                    slice_name=slice_name, slice_window=slice_window)
    west, _, east, _ = population.bounds
    wrap_x = math.isclose(east - west, 360, abs_tol=0.0001)
    report = {
        "slice": slice_name,
        "sourceWindow": window,
        "sourcePopulation": population.population,
        "romaniaCorrectionPeople": population.correction,
        "aggregationSeconds": monotonic() - start,
        "levels": [],
    }
    print(f"Prepared estimates for about {population.population + population.correction:,.0f} people "
          f"in {report['aggregationSeconds']:.1f} seconds.", flush=True)
    if population.correction:
        print(f"The Romania correction changed the source estimate by {population.correction:+,.0f} people.",
              flush=True)
    tiles_directory = staged / f"tiles_{tile_width}"
    manifest = {"version": 3, "tileSize": tile_width, "bounds": population.bounds, "levels": []}
    if raw:
        manifest.update(kind="raw", colourScale="log1p")
    name = area_title(slice_name) if slice_name != "world" else "the World"
    title = "Human Life, Pixel by Pixel"
    for split, (level, (counts, land)) in enumerate(zip(levels, population.grids)):
        metadata = level.metadata()
        if raw:
            print(f"Colouring the {level.width:,} by {level.height:,} map by estimated population.", flush=True)
            image, maximum = heatmap_image(counts, land)
            metadata.pop("peoplePerMark")
            metadata["maxPeoplePerPixel"] = maximum
            audit = {"aggregatedPopulation": float(counts.sum(dtype=np.float64))}
            subtitle = f"WorldPop population estimates · {name}"
            lines = [f"Log scale from 0 to {maximum:,.1f} people per pixel."]
        else:
            marks, audit = solve_level(level, counts, land, wrap_x, cache,
                                       cache_key(binary_key, level.width), solver_options)
            image = binary_image(land, marks)
            subtitle = f"One pixel for every {level.people_per_mark:,} people · {name}"
            lines = [f"{audit['marks']:,} bright pixels represent approximate locations of {audit['peopleRepresented']:,} people.",
                     "A diffusion algorithm spreads out dense areas and joins sparse areas to a single representative pixel."]
        tile_count = bytes_written = 0
        if tile_width is not None:
            tile_count, bytes_written = write_tiles(tiles_directory / str(split), image,
                                                   tile_width=tile_width, bits=8 if raw else 2)
            manifest["levels"].append({"split": split, **metadata})
            print(f"Saved {tile_count:,} tiles for the {level.width:,}-pixel map.", flush=True)
        if split == len(levels) - 1:
            cover = annotate(image, title, subtitle, lines, population.bounds) if annotated else image
            write_map(staged / MAP_FILENAME, cover, raw=raw)
        report["levels"].append({**metadata, **audit, "tiles": tile_count, "tileBytes": bytes_written})
    if tile_width is not None:
        write_json(tiles_directory / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE,
                        help="WorldPop population GeoTIFF (default: %(default)s)")
    parser.add_argument("output", nargs="?", type=Path,
                        help="output directory override (default: a named run under data/output/v3)")
    engine = parser.add_mutually_exclusive_group()
    engine.add_argument("--gpu", type=Path, default=DEFAULT_GPU,
                        help="path to the compiled Mojo executable (default: %(default)s)")
    engine.add_argument("--cpu", action="store_true", help="use NumPy instead of the GPU")
    parser.add_argument("--slice", dest="slice_name",
                        help=f"world, a named slice ({', '.join(WORLD_SLICES)}), or a name for --slice-idx")
    parser.add_argument("--slice-idx", dest="slice_window", type=int, nargs=4,
                        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"), help="custom half-open source window")
    parser.add_argument("--width", "--max-width", type=int, required=True, help="width of the map in pixels")
    parser.add_argument("--tile-width", type=int, help="also export tiles of this size at multiple zoom steps")
    parser.add_argument("--pop-target", type=int, required=True, help="people represented by one bright pixel")
    parser.add_argument("--romania-patch", action=argparse.BooleanOptionalAction, default=True,
                        help="correct the original WorldPop Romania window (default: enabled)")
    parser.add_argument("--annotated", action="store_true", help="draw a title and credits on the map")
    parser.add_argument("--raw", action="store_true", help="export a heatmap of the aggregated source population")
    parser.add_argument("--force-rerun", action="store_true", help="recompute and refresh cached data")
    parser.add_argument("--batch-passes", type=int, default=256)
    parser.add_argument("--max-passes", type=int, default=16_384)
    parser.add_argument("--overflow-tolerance", type=float, default=0.20,
                        help="allowed population above pixel capacity, in bright pixels' worth of people")
    parser.add_argument("--drift-tolerance", type=float, default=0.20,
                        help="allowed change in the population total, in bright pixels' worth of people")
    parser.add_argument("--report", type=Path, help="write population accounting and timing as JSON")
    args = vars(parser.parse_args())
    cpu = args.pop("cpu")
    if args["raw"] or cpu:
        args["gpu"] = None
    name = re.sub(r"\s+", "-", (args["slice_name"] or "").strip().lower())
    window = args["slice_window"]
    if window is not None:
        if not re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", name) or len(name) > 64:
            parser.error("--slice-idx requires a --slice name of up to 64 letters, digits, spaces, hyphens or underscores")
        if name in ("world", *WORLD_SLICES):
            parser.error(f"use a distinct name for the custom slice (\"{name}\" is reserved)")
        top, bottom, left, right = window
        if not (0 <= top < bottom and 0 <= left < right):
            parser.error("--slice-idx requires non-negative, increasing row and column indices")
    elif name and name not in ("world", *WORLD_SLICES):
        parser.error(f"unknown slice {name!r}; provide --slice-idx for a custom window")
    args["slice_name"] = name or "world"
    tile_width = args["tile_width"]
    width = args.pop("width")
    pop_target = args.pop("pop_target")
    report_path = args.pop("report")
    if args["output"] is None:
        prefix = "raw_" if args["raw"] else ""
        run_name = f"{prefix}{args['slice_name']}_{width}_{pop_target}"
        args["output"] = Path("data/output/v3") / run_name

    output = args["output"].parent.resolve() / args["output"].name
    mode = "population heatmap" if args["raw"] else "binary population map"
    processor = (" on the GPU" if args["gpu"] else " on the CPU") if not args["raw"] else ""
    print(f"Making a {width:,}-pixel-wide {mode} for {area_title(args['slice_name'])}{processor}.", flush=True)
    print(f"Output will be published to {output}", flush=True)
    try:
        gpu = args["gpu"]
        if gpu is not None and (not gpu.is_file() or not os.access(gpu, os.X_OK)):
            raise ValueError(f"GPU executable is missing or not executable at {gpu}. "
                             "Run 'pixi run --locked build-gpu' to build the default, or use --cpu.")
        args["levels"] = levels_for_source(args["source"], width, pop_target, tile_width=tile_width,
                                           slice_name=args["slice_name"], slice_window=window)
        generate(**args, cache_dir=Path("data/cache/v3"), report_path=report_path)
    except (ValueError, OSError, CalledProcessError) as error:
        parser.exit(1, f"Generation failed for {output}\n{error}\n")
    print(f"Saved {output / MAP_FILENAME}", flush=True)
