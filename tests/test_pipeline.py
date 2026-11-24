"""Population accounting checks on small fixtures with known answers."""

import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
from PIL import Image
from tifffile import imwrite

from poppyn.population import Bins, Level, correct_romania, read_population
from poppyn.prepare import generate, pinholes
from poppyn.redistribution import child_blocks, gather, round_marks, settle
from poppyn.render import binary_image, write_level


def write_population_tiff(path, counts, nodata="-9999"):
    imwrite(path, counts, tile=(16, 16), metadata=None, extratags=[
        (33550, "d", 3, (0.25, 0.25, 0), False),
        (33922, "d", 6, (0, 0, 0, 12, 33, 0), False),
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

    def test_file_metadata_nodata_and_publish(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "population.tif"
            output = parent / "tiles"
            output.mkdir()
            nodata = np.finfo(np.float32).min
            raw = np.full((4, 5), nodata, dtype=np.float32)
            raw[0, 0], raw[1, 0], raw[2, 4] = 0, 30, 10
            write_population_tiff(source, raw, str(float(nodata)))
            level = Level(2, 2, 40)
            population = read_population(source, [level])
            counts, land = population.grids[0]
            self.assertEqual(population.bounds, (12, 32, 13.25, 33))
            self.assertEqual(population.population, 40)
            np.testing.assert_array_equal(land, [[True, False], [False, True]])
            np.testing.assert_array_equal(counts, [[30, 0], [0, 10]])
            generate(source, output, [level], tile_width=2)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["levels"], [level.metadata()])
            with Image.open(output / "2/0/0.png") as tile, Image.open(output / "2/map.png") as image:
                np.testing.assert_array_equal(np.array(tile), [[2, 0], [0, 1]])
                np.testing.assert_array_equal(image, tile)

    def test_romania_patch_keeps_original_mask(self):
        values = np.array([[15, -9999], [0, 25]], dtype=np.float32)
        values.flags.writeable = False
        changed, delta = correct_romania(values, values >= 0, 4738, 24755)
        np.testing.assert_array_equal(changed, [[10, -9999], [10, 10]])
        self.assertEqual(delta, -10)

    def test_named_slice_crosses_tiff_tile_edges(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "population.tif"
            raw = np.full((18, 20), 1000, dtype=np.float32)
            raw[14:18, 14:19] = np.arange(1, 21).reshape(4, 5)
            raw[15, 15] = -9999
            write_population_tiff(source, raw)
            with patch("poppyn.population.WORLD_SHAPE", raw.shape), \
                    patch.dict("poppyn.population.WORLD_SLICES", {"fixture": (14, 18, 14, 19)}):
                population = read_population(source, [Level(2, 2, 50)], slice_name="fixture")
            counts, land = population.grids[0]
            self.assertEqual(population.bounds, (15.5, 28.5, 16.75, 29.5))
            self.assertEqual(population.population, 203)
            np.testing.assert_array_equal(counts, [[20, 28], [87, 68]])
            self.assertTrue(land.all())

    def test_romania_correction_is_enabled_by_default_with_absolute_slice_indices(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "population.tif"
            counts = np.full((4, 5), 25, dtype=np.float32)
            counts[1, 2] = -9999
            write_population_tiff(source, counts)
            with patch("poppyn.population.WORLD_SHAPE", counts.shape), \
                    patch("poppyn.population.WORLD_BOUNDS", (12, 32, 13.25, 33)), \
                    patch("poppyn.population.ROMANIA", (1, 3, 2, 4)):
                options = {"slice_name": "fixture", "slice_window": (1, 3, 2, 4)}
                levels = [Level(2, 2, 25)]
                corrected = read_population(source, levels, **options)
                original = read_population(source, levels, romania_patch=False, **options)
            self.assertEqual(corrected.correction, -45)
            self.assertEqual(original.correction, 0)
            np.testing.assert_array_equal(corrected.grids[0][0], [[0, 10], [10, 10]])
            np.testing.assert_array_equal(original.grids[0][0], [[0, 25], [25, 25]])

    def test_png_edge_tile_dimensions(self):
        level = Level(513, 515, 1)
        land = np.zeros((level.height, level.width), dtype=bool)
        marks = np.zeros_like(land)
        land[-1, -1] = marks[-1, -1] = True
        with TemporaryDirectory() as directory:
            output = Path(directory)
            count, _ = write_level(output, level, binary_image(land, marks), tile_width=512, bits=2)
            self.assertEqual(count, 4)
            with Image.open(output / "513/1/1.png") as image:
                self.assertEqual(image.size, (1, 3))
                self.assertEqual(image.getpixel((0, 2)), 2)


class OutputTests(TestCase):
    def test_cli_names_maps_by_width_and_includes_an_exact_maximum(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/raw/ppp_2020_1km_Aggregated.tif"
            source.parent.mkdir(parents=True)
            write_population_tiff(source, np.ones((8, 10), dtype=np.float32))
            script = Path(__file__).resolve().parents[1] / "run_all.py"
            command = [sys.executable, "-B", str(script),
                       "--slice", "Test Region", "--slice-idx", "1", "7", "1", "10",
                       "--tile-width", "4", "--max-width", "9", "--pop-target", "100", "--annotated"]
            result = subprocess.run([*command, "--cpu"], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = root.resolve() / "data/output/v3/test-region_4_9_100/tiles"
            self.assertIn(f"Maps published to {output}", result.stdout)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["version"], 2)
            self.assertEqual(manifest["tileSize"], 4)
            self.assertEqual(manifest["levels"], [Level(4, 3, 450).metadata(),
                                                  Level(8, 5, 135).metadata(),
                                                  Level(9, 6, 100).metadata()])
            self.assertEqual({path.name for path in output.iterdir() if path.is_dir()}, {"4", "8", "9"})
            self.assertTrue((output / "9/map-annotated.png").is_file())
            with Image.open(output / "9/2/1.png") as tile:
                self.assertEqual(tile.size, (1, 2))

            result = subprocess.run([*command, "--raw"], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            raw_run = output.parent.parent / "raw_test-region_4_9_100"
            self.assertTrue((raw_run / "report.json").is_file())
            self.assertTrue((raw_run / "tiles/9/map-annotated.png").is_file())
            raw_manifest = json.loads((raw_run / "tiles/manifest.json").read_text())
            self.assertEqual(raw_manifest["kind"], "raw")

            result = subprocess.run([*command, "--raw", str(source), str(output)],
                                    cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already contains binary maps", result.stderr)
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)

    def test_raw_map_tiles_and_annotation_share_the_same_heatmap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "population.tif"
            output = root / "raw"
            write_population_tiff(source, np.array([[0, 1, 5], [-9999, 10, 100]], dtype=np.float32))
            with patch("poppyn.prepare.settle", side_effect=AssertionError("raw export ran the solver")):
                generate(source, output, [Level(3, 2, 50)], tile_width=3, raw=True, annotated=True)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["colourScale"], "log1p")
            self.assertEqual(manifest["levels"][0]["maxPeoplePerPixel"], 100)
            with Image.open(output / "3/map.png") as image:
                pixels = np.asarray(image)
                self.assertEqual(pixels[1, 0], 0)
                self.assertEqual(pixels[1, 2], 255)
                self.assertEqual(np.unique(pixels).size, 6)
                with Image.open(output / "3/0/0.png") as tile:
                    np.testing.assert_array_equal(image, tile)
                with Image.open(output / "3/map-annotated.png") as captioned:
                    offset = (captioned.width - image.width) // 2
                    np.testing.assert_array_equal(captioned.crop((offset, 0, offset + 3, 2)), image.convert("RGB"))

    def test_cached_rerun_reuses_results_and_source_changes_invalidate_them(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "population.tif"
            output = root / "tiles"
            levels = [Level(2, 2, 20)]
            options = {"cache_dir": root / "cache", "tile_width": 2}
            write_population_tiff(source, np.full((2, 2), 5, dtype=np.float32))
            generate(source, output, levels, **options)
            original_bytes = (output / "2/map.png").read_bytes()
            with patch("poppyn.prepare.read_population", side_effect=AssertionError("re-read source")), \
                    patch("poppyn.prepare.settle", side_effect=AssertionError("repeated settlement")):
                generate(source, output, levels, annotated=True, **options)
            self.assertEqual((output / "2/map.png").read_bytes(), original_bytes)
            self.assertTrue((output / "2/map-annotated.png").is_file())
            with patch("poppyn.prepare.read_population", wraps=read_population) as reader, \
                    patch("poppyn.prepare.settle", wraps=settle) as solver:
                generate(source, output, levels, force_rerun=True, **options)
                reader.assert_called_once()
                solver.assert_called_once()
            write_population_tiff(source, np.full((2, 2), 10, dtype=np.float32))
            report = generate(source, output, levels, **options)
            self.assertEqual(report["levels"][0]["marks"], 2)
            self.assertNotEqual((output / "2/map.png").read_bytes(), original_bytes)

            latest_bytes = (output / "2/map.png").read_bytes()
            write_population_tiff(source, np.full((2, 2), 15, dtype=np.float32))
            with self.assertRaises(FileNotFoundError):
                generate(source, output, levels, report_path=root / "missing" / "report.json", **options)
            self.assertEqual((output / "2/map.png").read_bytes(), latest_bytes)

            unrelated = output / "notes.txt"
            unrelated.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the catalogue layout"):
                generate(source, output, levels, **options)
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")


class RedistributionTests(TestCase):
    def test_gather_reflects_at_land_mask_and_domain_edges(self):
        start = np.array([[4.0, 0], [-1, 0]])
        expected = np.array([[3.25, 0.75], [-1, 0]])
        np.testing.assert_array_equal(gather(start), expected)
        self.assertEqual(expected[expected >= 0].sum(), start[start >= 0].sum())

    def test_gather_can_wrap_across_longitude(self):
        start = np.array([[3.0, 0, 0]])
        np.testing.assert_array_equal(gather(start), [[2.5, 0.5, 0]])
        np.testing.assert_array_equal(gather(start, wrap_x=True), [[2, 0.5, 0.5]])

    def test_failed_island_convergence_does_not_publish(self):
        counts = np.array([[0.0, 0, 3]])
        land = np.array([[True, False, True]])
        with self.assertRaisesRegex(ValueError, "eligible cells cannot represent"):
            settle(counts, land, 1)
        counts[0, 2] = 1.75
        with self.assertRaisesRegex(ValueError, "overfull land component"):
            settle(counts, land, 1, batch_passes=1, max_passes=2)

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
            self.assertEqual(pinholes(land, marks), 0)
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
        while counts.shape != (1, 1):
            counts, marks = (child_blocks(grid).sum(axis=(2, 3)) for grid in (counts, marks))
            self.assertTrue(((marks >= np.floor(counts)) & (marks <= np.ceil(counts))).all())


if __name__ == "__main__":
    main()
