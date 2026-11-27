"""Population accounting checks on small fixtures with known answers."""

from concurrent.futures import ThreadPoolExecutor
from fcntl import LOCK_NB, flock
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, main_thread, current_thread
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
from PIL import Image
from tifffile import imwrite

from poppyn.output import catalogue_stage
from poppyn.population import Bins, Level, levels_for_source, read_population
from poppyn.prepare import generate
from poppyn.projection import EQUAL_AREA, EQUAL_EARTH, GEOGRAPHIC, equal_earth_parallel
from poppyn.redistribution import gather, round_marks, settle
from poppyn.render import LAND, MARK, OUTSIDE, WATER, annotate, binary_image, write_tiles


def write_population_tiff(path, counts, nodata="-9999", *, step=0.25, west=12, north=33):
    imwrite(path, counts, tile=(16, 16), metadata=None, extratags=[
        (33550, "d", 3, (step, step, 0), False),
        (33922, "d", 6, (0, 0, 0, west, north, 0), False),
        (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326), False),
        (42113, "s", 0, nodata, False),
    ])


class PopulationTests(TestCase):
    def test_bin_boundaries_and_split_source(self):
        values = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        bins = Bins(values.shape, (2, 2))
        # The TIFF boundary does not have to line up with the aggregation grid.
        bins.add(values[:1], np.ones((1, 4), dtype=bool), 0, 0)
        bins.add(values[1:], np.ones((3, 4), dtype=bool), 1, 0)
        np.testing.assert_array_equal(bins.counts, [[14, 22], [46, 54]])
        self.assertEqual(bins.counts.sum(), values.sum())

    def test_uniform_population_stays_uniform_at_fractional_scale(self):
        values = np.full((7, 5), 6, dtype=np.float32)
        bins = Bins(values.shape, (3, 2))
        # Both chunk boundaries follow source cells shared by adjacent output cells.
        for top, bottom in ((0, 3), (3, 7)):
            for left, right in ((0, 3), (3, 5)):
                chunk = values[top:bottom, left:right]
                bins.add(chunk, np.ones(chunk.shape, dtype=bool), top, left)
        np.testing.assert_allclose(bins.counts, np.full((3, 2), 35), rtol=0, atol=1e-12)
        self.assertAlmostEqual(bins.counts.sum(), values.sum())
        self.assertTrue(bins.land.all())

    def test_centred_source_cell_shares_equally_across_four_output_cells(self):
        values = np.full((5, 5), -9999, dtype=np.float32)
        for population in (0, 40):
            with self.subTest(population=population):
                values[2, 2] = population
                bins = Bins(values.shape, (2, 2))
                bins.add(values, values >= 0, 0, 0)
                np.testing.assert_array_equal(bins.counts, np.full((2, 2), population / 4))
                self.assertTrue(bins.land.all())

    def test_equal_area_regridding_preserves_population_and_nodata(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "population.tif"
            values = np.full((16, 16), -9999, dtype=np.float32)
            values[:, 5] = 0
            values[8, 5] = 400
            write_population_tiff(source, values)
            levels = levels_for_source(source, 16, 200, slice_name="world", slice_window=None,
                                       projection="equal-area")
            counts, land = read_population(source, levels, projection="equal-area").grids[0]
            self.assertEqual(counts.shape, (18, 16))
            self.assertAlmostEqual(counts.sum(), 400)
            np.testing.assert_array_equal(np.flatnonzero(land.any(axis=0)), [5])
            self.assertTrue(land[:, 5].all())
            rows, columns = counts.nonzero()
            np.testing.assert_array_equal(rows, [8, 9, 10])
            np.testing.assert_array_equal(columns, [5, 5, 5])

    def test_equal_earth_matches_authors_reference_coordinates(self):
        # Published WGS 84 example with a central meridian of 90° W.
        latitude = 34 + 3 / 60 + 27.169 / 3600
        longitude_delta = 90 - (117 + 11 / 60 + 48.349 / 3600)
        northing, longitude_scale = equal_earth_parallel(latitude)
        self.assertAlmostEqual(northing, 4_242_849.758, places=2)
        self.assertAlmostEqual(longitude_delta * longitude_scale, -2_390_749.043, places=2)

    def test_lake_mask_keeps_shores_and_islands_across_source_tile_and_crop_edges(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "population.tif"
            write_population_tiff(source, np.full((32, 32), 100, dtype=np.float32))
            mask_data = root / "lakes.geojson.gz"
            with gzip.open(mask_data, "wt", encoding="utf-8") as file:
                json.dump({"features": [{"geometry": {"type": "Polygon", "coordinates": [
                    [[14, 27], [18, 27], [18, 31], [14, 31], [14, 27]],
                    [[15.75, 28.75], [15.75, 29.25], [16.25, 29.25], [16.25, 28.75], [15.75, 28.75]],
                ]}}]}, file)
            with patch("poppyn.lakes.LAKES_PATH", mask_data):
                full = read_population(source, [Level(32, 32, 100)])
                crop = read_population(source, [Level(12, 8, 100)], slice_name="fixture",
                                       slice_window=(12, 20, 10, 22))
                levels = levels_for_source(source, 32, 100, slice_name="world", slice_window=None,
                                           projection=EQUAL_EARTH)
                report = generate(source, root / "out", levels, projection=EQUAL_EARTH, raw=True,
                                  cache_dir=root / "cache")
                cached = generate(source, root / "out", levels, projection=EQUAL_EARTH, raw=True,
                                  cache_dir=root / "cache")

            counts, land = full.grids[0]
            # The 16-by-16 lake leaves a 12-by-12 interior. Its island and margin occupy 6 by 6.
            self.assertEqual(full.lake_removed, (12**2 - 6**2) * 100)
            self.assertEqual(full.population - counts.sum(), full.lake_removed)
            self.assertFalse(counts[~land].any())
            self.assertTrue(land[9, 14])  # Shoreline band.
            self.assertTrue(land[13, 13])  # Diagonal neighbour of the island's shoreline.
            self.assertTrue(land[16, 16])  # Island at the junction of four tiles.
            self.assertFalse(land[16, 19])  # Open water immediately beyond the island's band.
            np.testing.assert_array_equal(crop.grids[0][0], counts[12:20, 10:22])
            np.testing.assert_array_equal(crop.grids[0][1], land[12:20, 10:22])
            self.assertAlmostEqual(report["lakePopulationRemovedPeople"], full.lake_removed)
            self.assertAlmostEqual(report["levels"][0]["aggregatedPopulation"], counts.sum())
            self.assertEqual(cached["lakePopulationRemovedPeople"], report["lakePopulationRemovedPeople"])

    def test_equal_earth_slice_includes_extra_population_across_antimeridian(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "population.tif"
            values = np.full((16, 180), -9999, dtype=np.float32)
            values[0, 179] = 100  # Within the original slice.
            values[0, 0] = 50  # Included across the antimeridian.
            values[0, 1] = 100  # Partially within the projected frame.
            values[0, 5] = 1000  # Beyond the projected frame.
            values[15, 179] = 1000  # Outside the latitude window.
            write_population_tiff(source, values, step=2, west=-180, north=80)
            options = {"slice_name": "fixture", "slice_window": (0, 14, 168, 180), "projection": EQUAL_EARTH}
            levels = levels_for_source(source, 12, 100, tile_width=5, **options)
            population = read_population(source, levels, **options)
            self.assertEqual(population.bounds, (156, 52, 180, 80))
            self.assertGreater(population.population, 150)
            self.assertLess(population.population, 250)
            for counts, land in population.grids:
                self.assertAlmostEqual(counts.sum(), population.population)
                self.assertFalse(counts[~land].any())

    def test_file_metadata_nodata_and_publish(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "population.tif"
            output = parent / "tiles"
            output.mkdir()
            nodata = np.finfo(np.float32).min
            raw = np.full((4, 5), nodata, dtype=np.float32)
            raw[0, 0], raw[1, 0], raw[2, 4] = 0, 30, 10
            write_population_tiff(source, raw, str(float(nodata)), west=-30, north=-5)
            level = Level(2, 2, 40)
            population = read_population(source, [level])
            counts, land = population.grids[0]
            self.assertEqual(population.bounds, (-30, -6, -28.75, -5))
            self.assertEqual(population.population, 40)
            np.testing.assert_array_equal(land, [[True, False], [False, True]])
            np.testing.assert_array_equal(counts, [[30, 0], [0, 10]])
            generate(source, output, [level])
            self.assertEqual({path.name for path in output.iterdir()}, {"poppyn-map.png"})
            with Image.open(output / "poppyn-map.png") as image:
                np.testing.assert_array_equal(image, [[2, 0], [0, 1]])

    def test_named_slice_crosses_tiff_tile_edges(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "population.tif"
            raw = np.full((18, 20), 1000, dtype=np.float32)
            raw[14:18, 14:19] = np.arange(1, 21).reshape(4, 5)
            raw[15, 15] = -9999
            write_population_tiff(source, raw)
            with patch.dict("poppyn.population.WORLD_SLICES", {"fixture": (15.55, 28.51, 16.74, 29.45)}):
                population = read_population(source, [Level(2, 2, 50)], slice_name="fixture")
            counts, land = population.grids[0]
            self.assertEqual(population.bounds, (15.5, 28.5, 16.75, 29.5))
            self.assertEqual(population.window, (14, 18, 14, 19))
            self.assertEqual(population.population, 203)
            np.testing.assert_array_equal(counts, [[14.5, 33.5], [71.5, 83.5]])
            self.assertTrue(land.all())

    def test_png_edge_tile_dimensions(self):
        level = Level(513, 515, 1)
        land = np.zeros((level.height, level.width), dtype=bool)
        marks = np.zeros_like(land)
        land[-1, -1] = marks[-1, -1] = True
        with TemporaryDirectory() as directory:
            output = Path(directory)
            count, _ = write_tiles(output / "0", binary_image(land, marks), tile_width=512, bits=2)
            self.assertEqual(count, 4)
            with Image.open(output / "0/1/1.png") as image:
                self.assertEqual(image.size, (1, 3))
                self.assertEqual(image.getpixel((0, 2)), 2)


class OutputTests(TestCase):
    def test_nodata_land_has_land_colour_and_never_receives_a_mark(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "population.tif"
            values = np.full((16, 16), -9999, dtype=np.float32)
            values[4, 4] = 100
            write_population_tiff(source, values, step=1, west=-8, north=8)
            land_path, lake_path = (root / f"{name}.geojson.gz" for name in ("land", "lake"))
            for path, extent in ((land_path, 6), (lake_path, 2)):
                ring = [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent], [-extent, -extent]]
                with gzip.open(path, "wt", encoding="utf-8") as file:
                    json.dump({"features": [{"geometry": {"type": "Polygon", "coordinates": [ring]}}]}, file)
            with patch("poppyn.basemap.LAND_PATH", land_path), patch("poppyn.lakes.LAKES_PATH", lake_path):
                for projection in (GEOGRAPHIC, EQUAL_AREA, EQUAL_EARTH):
                    levels = levels_for_source(source, 16, 100, slice_name="world", slice_window=None,
                                               projection=projection)
                    for raw in (False, True):
                        with self.subTest(projection=projection, raw=raw):
                            output = root / f"{projection}-{raw}"
                            report = generate(source, output, levels, projection=projection, raw=raw)
                            self.assertAlmostEqual(report["levels"][0]["aggregatedPopulation"], 100)
                            with Image.open(output / "poppyn-map.png") as image:
                                colours = image.convert("RGB")
                                middle = image.height // 2
                                self.assertEqual(colours.getpixel((0, middle)), WATER)  # Ocean.
                                self.assertEqual(colours.getpixel((3, middle)), LAND)   # NoData on land.
                                self.assertEqual(colours.getpixel((8, middle)), WATER)  # Lake.
                                if not raw:
                                    bright = np.all(np.asarray(colours) == MARK, axis=-1)
                                    self.assertEqual(bright.sum(), 1)
                                    self.assertLess(np.nonzero(bright)[0][0], middle)
                                    self.assertLess(np.nonzero(bright)[1][0], 6)

    def test_equal_earth_world_distinguishes_outside_from_water(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "world.tif"
            write_population_tiff(source, np.ones((24, 72), dtype=np.float32), step=5, west=-180, north=60)
            levels = levels_for_source(source, 48, 100, tile_width=16, slice_name="world", slice_window=None,
                                       projection=EQUAL_EARTH)
            for raw in (False, True):
                with self.subTest(raw=raw):
                    output = root / ("raw" if raw else "binary")
                    report = generate(source, output, levels, tile_width=16, projection=EQUAL_EARTH, raw=raw)
                    self.assertAlmostEqual(report["sourcePopulation"], 1728)
                    self.assertEqual(report["centralMeridian"], 0)
                    self.assertAlmostEqual(report["levels"][-1]["aggregatedPopulation"], 1728)
                    with Image.open(output / "poppyn-map.png") as image:
                        colours = image.convert("RGB")
                        self.assertEqual(colours.getpixel((0, 0)), OUTSIDE)
                        self.assertEqual(colours.getpixel((image.width // 2, 0)), WATER)
                    with Image.open(output / "tiles_16/2/0/0.png") as tile:
                        self.assertEqual(tile.convert("RGB").getpixel((0, 0)), OUTSIDE)

    def test_cli_names_maps_by_width_and_includes_an_exact_maximum(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/raw/global_pop_2025_CN_1km_R2025A_v1.tif"
            source.parent.mkdir(parents=True)
            write_population_tiff(source, np.ones((8, 10), dtype=np.float32))
            script = Path(__file__).resolve().parents[1] / "run_all.py"
            command = [sys.executable, "-B", str(script),
                       "--slice", "Test Region", "--slice-idx", "1", "7", "1", "10",
                       "--tile-width", "4", "--width", "9", "--pop-target", "100"]
            result = subprocess.run([*command, "--cpu"], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = root.resolve() / "data/output/v3/test-region_9_100"
            map_path = output / "poppyn-map.png"
            with Image.open(map_path) as image:
                self.assertEqual(image.size, (9, 6))
                pixels = np.asarray(image.convert("RGB"))
            tiles = output / "tiles_4"
            manifest = json.loads((tiles / "manifest.json").read_text())
            self.assertEqual(manifest["version"], 3)
            self.assertEqual(manifest["tileSize"], 4)
            self.assertEqual(
                [(entry["split"], entry["width"], entry["height"], entry["peoplePerMark"])
                 for entry in manifest["levels"]],
                [(0, 4, 3, 450), (1, 8, 5, 135), (2, 9, 6, 100)],
            )
            self.assertEqual({path.name for path in tiles.iterdir() if path.is_dir()}, {"0", "1", "2"})
            self.assertEqual(list(tiles.rglob("map*.png")), [])
            covered = np.zeros(pixels.shape[:2], dtype=bool)
            for path in (tiles / "2").glob("*/*.png"):
                y, x = int(path.stem) * 4, int(path.parent.name) * 4
                with Image.open(path) as tile:
                    self.assertEqual(tile.size, (min(4, 9 - x), min(4, 6 - y)))
                    window = np.s_[y:y + tile.height, x:x + tile.width]
                    self.assertFalse(covered[window].any())
                    np.testing.assert_array_equal(tile.convert("RGB"), pixels[window])
                    covered[window] = True
            self.assertTrue(covered.all())

            result = subprocess.run([*command, "--raw"], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            raw_run = output.parent / "raw_test-region_9_100"
            self.assertTrue((raw_run / "poppyn-map.png").is_file())
            raw_manifest = json.loads((raw_run / "tiles_4/manifest.json").read_text())
            self.assertEqual(raw_manifest["kind"], "raw")

            original_map = map_path.read_bytes()
            report = root / "projected.json"
            result = subprocess.run([*command, "--cpu", "--projection", "equal-area", "--report", str(report)],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            projected = output.parent / "test-region_9_100_equal-area"
            with Image.open(projected / "poppyn-map.png") as image:
                self.assertEqual(image.size, (9, 7))
            projected_manifest = json.loads((projected / "tiles_4/manifest.json").read_text())
            self.assertEqual(projected_manifest["crs"], "EPSG:6933")
            self.assertEqual(projected_manifest["bounds"], manifest["bounds"])
            projected_report = json.loads(report.read_text())
            self.assertAlmostEqual(projected_report["levels"][-1]["aggregatedPopulation"], 54)
            self.assertEqual(projected_report["levels"][-1]["marks"], 1)
            self.assertEqual(map_path.read_bytes(), original_map)

            result = subprocess.run([*command, "--raw", str(source), str(output)],
                                    cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(map_path.read_bytes(), original_map)
            self.assertEqual(json.loads((tiles / "manifest.json").read_text()), manifest)

    def test_raw_map_and_tiles_share_the_same_heatmap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "population.tif"
            output = root / "raw"
            write_population_tiff(source, np.array([[0, 1, 5], [-9999, 10, 100]], dtype=np.float32),
                                  west=-30, north=-5)
            with patch("poppyn.prepare.settle", side_effect=AssertionError("raw export ran the solver")):
                generate(source, output, [Level(3, 2, 50)], tile_width=3, raw=True)
            manifest = json.loads((output / "tiles_3/manifest.json").read_text())
            self.assertEqual(manifest["colourScale"], "log1p")
            self.assertEqual(manifest["levels"][0]["maxPeoplePerPixel"], 100)
            with Image.open(output / "poppyn-map.png") as image:
                pixels = np.asarray(image)
                self.assertEqual(pixels[1, 0], 0)
                self.assertEqual(pixels[1, 2], 255)
                self.assertEqual(np.unique(pixels).size, 6)
                with Image.open(output / "tiles_3/0/0/0.png") as tile:
                    np.testing.assert_array_equal(image, tile)


    def test_annotation_keeps_underlying_map_visible(self):
        captions = ("World population", "One bright pixel per 62,500 people",
                    ["Dense areas spread out and sparse populations share a pixel."],
                    (-180, -72, 180, 84))
        darker = annotate(Image.new("RGB", (480, 208), (0, 0, 0)), *captions)
        lighter = annotate(Image.new("RGB", (480, 208), (80, 80, 80)), *captions)
        self.assertEqual(darker.size, (480, 208))
        self.assertEqual(lighter.size, darker.size)
        dark_red = np.asarray(darker, dtype=int)[:, :, 0]
        light_red = np.asarray(lighter, dtype=int)[:, :, 0]
        title = dark_red[:50]
        strongest_ink = title == title.max()
        self.assertGreater(title.max(), 0)
        # Even in the solid parts of the lettering, some background must show through.
        self.assertTrue((light_red[:50][strongest_ink] > title[strongest_ink]).all())
        self.assertTrue((light_red[:50][strongest_ink] - title[strongest_ink] < 80).all())
        self.assertEqual(lighter.getpixel((240, 104)), (80, 80, 80))

    def test_rerun_accepts_an_export_above_pillows_pixel_limit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "population.tif"
            output = root / "map"
            levels = [Level(2, 2, 20)]
            write_population_tiff(source, np.full((2, 2), 5, dtype=np.float32))
            generate(source, output, levels)

            write_population_tiff(source, np.full((2, 2), 10, dtype=np.float32))
            with patch.object(Image, "MAX_IMAGE_PIXELS", 1):
                with self.assertRaises(Image.DecompressionBombError):
                    Image.open(output / "poppyn-map.png")
                report = generate(source, output, levels)

            self.assertEqual(report["levels"][0]["marks"], 2)
            with Image.open(output / "poppyn-map.png") as image:
                self.assertEqual(np.count_nonzero(np.asarray(image) == 2), 2)

    def test_waiting_export_retries_when_the_lock_file_is_replaced(self):
        opened_old_lock, proceed, blocked_on_current = Event(), Event(), Event()
        attempts = 0

        def acquire(fd, operation):
            nonlocal attempts
            if current_thread() is not main_thread():
                attempts += 1
                if attempts == 1:
                    opened_old_lock.set()
                    self.assertTrue(proceed.wait(timeout=5))
                elif attempts == 2:
                    with self.assertRaises(BlockingIOError):
                        flock(fd, operation | LOCK_NB)
                    blocked_on_current.set()
            flock(fd, operation)

        with TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=1) as worker:
            output = Path(directory) / "map"
            lock = output.with_name(".map.lock")

            def stage_again():
                with catalogue_stage(output, None, raw=False):
                    self.assertTrue(lock.is_file())

            with patch("poppyn.output.fcntl.flock", side_effect=acquire):
                with catalogue_stage(output, None, raw=False):
                    waiting = worker.submit(stage_again)
                    self.assertTrue(opened_old_lock.wait(timeout=5))
                self.assertFalse(lock.exists())
                with catalogue_stage(output, None, raw=False):
                    proceed.set()
                    self.assertTrue(blocked_on_current.wait(timeout=5))
                    self.assertFalse(waiting.done())
                waiting.result(timeout=5)
            self.assertFalse(lock.exists())

    def test_cached_rerun_reuses_results_and_source_changes_invalidate_them(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "population.tif"
            output = root / "tiles"
            levels = [Level(2, 2, 20)]
            options = {"cache_dir": root / "cache", "tile_width": 2}
            write_population_tiff(source, np.full((2, 2), 5, dtype=np.float32))
            generate(source, output, levels, **options)
            original_bytes = (output / "poppyn-map.png").read_bytes()
            with patch("poppyn.prepare.read_population", side_effect=AssertionError("re-read source")), \
                    patch("poppyn.prepare.settle", side_effect=AssertionError("repeated settlement")):
                generate(source, output, levels, **options)
            self.assertEqual((output / "poppyn-map.png").read_bytes(), original_bytes)
            report = generate(source, output, levels, force_rerun=True, **options)
            self.assertFalse(report["levels"][0]["redistributionCacheHit"])
            self.assertEqual((output / "poppyn-map.png").read_bytes(), original_bytes)
            write_population_tiff(source, np.full((2, 2), 10, dtype=np.float32))
            report = generate(source, output, levels, **options)
            self.assertEqual(report["levels"][0]["marks"], 2)
            self.assertNotEqual((output / "poppyn-map.png").read_bytes(), original_bytes)

            latest_bytes = (output / "poppyn-map.png").read_bytes()
            write_population_tiff(source, np.full((2, 2), 15, dtype=np.float32))
            with self.assertRaises(FileNotFoundError):
                generate(source, output, levels, report_path=root / "missing" / "report.json", **options)
            self.assertEqual((output / "poppyn-map.png").read_bytes(), latest_bytes)
            self.assertFalse((root / ".tiles.lock").exists())

            unrelated = output / "notes.txt"
            unrelated.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the catalogue layout"):
                generate(source, output, levels, **options)
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")


class RedistributionTests(TestCase):
    def test_gather_reflects_at_land_mask_and_domain_edges(self):
        start = np.array([[4.0, 0], [-1, 0]])
        expected = np.array([[3.25, 0.75], [-1, 0]])
        result = gather(start)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result[result >= 0].sum(), start[start >= 0].sum())

    def test_gather_can_wrap_across_longitude(self):
        start = np.array([[3.0, 0, 0]])
        np.testing.assert_array_equal(gather(start), [[2.5, 0.5, 0]])
        np.testing.assert_array_equal(gather(start, wrap_x=True), [[2, 0.5, 0.5]])

    def test_settlement_rejects_insufficient_global_or_island_capacity(self):
        counts = np.array([[0.0, 0, 3]])
        land = np.array([[True, False, True]])
        with self.assertRaisesRegex(ValueError, "eligible cells cannot represent"):
            settle(counts, land, 1)
        counts[0, 2] = 1.75
        with self.assertRaisesRegex(ValueError, "stalled.*overfull land component"):
            settle(counts, land, 1)

    def test_settlement_can_continue_past_an_explicit_pass_limit(self):
        counts = np.zeros((1, 12))
        counts[0, 6] = 5
        land = np.ones(counts.shape, dtype=bool)
        progress = []
        with self.assertRaisesRegex(ValueError, "requested limit of 3 rounds"):
            settle(counts, land, 1, batch_passes=2, max_passes=3,
                   progress=lambda passes, overflow, drift: progress.append((passes, overflow)))
        self.assertEqual(progress[-1][0], 3)
        self.assertLess(progress[-1][1], progress[0][1])

        result = settle(counts, land, 1, batch_passes=2)
        self.assertGreater(result.passes, 3)
        self.assertEqual(round_marks(result.grid, land, 5).sum(), 5)

    def test_spike_forms_one_translatable_filled_blob(self):
        grids = []
        for row, col in ((14, 16), (18, 19)):
            counts = np.zeros((35, 37))
            counts[row, col] = 50
            land = np.ones_like(counts, dtype=bool)
            result = settle(counts, land, 1, batch_passes=128, max_passes=512)
            self.assertLess(abs(result.mass_drift), 1e-9)
            marks = round_marks(result.grid, land, 50)
            self.assertEqual(marks.sum(), 50)
            neighbourhoods = np.lib.stride_tricks.sliding_window_view(marks, (3, 3))
            self.assertFalse(((neighbourhoods.sum(axis=(2, 3)) == 8) & ~marks[1:-1, 1:-1]).any())
            grids.append(result.grid[row - 8:row + 9, col - 8:col + 9])
        np.testing.assert_allclose(*grids, rtol=0, atol=1e-10)

    def test_rounding_preserves_budget_and_local_bounds(self):
        rng = np.random.default_rng(41)
        counts = rng.random((19, 13))
        counts[2, 3] = 1
        counts[3, 2] = 0
        land = np.ones_like(counts, dtype=bool)
        land[2, 2] = False
        counts[~land] = 0
        budget = math.floor(counts.sum() + 0.5)
        marks = round_marks(counts, land, budget)
        np.testing.assert_array_equal(marks, round_marks(counts, land, budget))
        self.assertEqual(marks.sum(), budget)
        self.assertTrue(marks[2, 3])
        self.assertFalse(marks[3, 2])
        self.assertFalse(marks[2, 2])
        height, width = counts.shape
        for size in (2, 4, 8, 16, 32):
            for y in range(0, height, size):
                for x in range(0, width, size):
                    window = np.s_[y:y + size, x:x + size]
                    population = counts[window].sum()
                    self.assertIn(marks[window].sum(), (math.floor(population), math.ceil(population)))


if __name__ == "__main__":
    main()
