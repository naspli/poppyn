"""Read WorldPop counts into aligned, population-preserving display grids."""

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path

import numpy as np
from tifffile import TiffFile

from .statics import WORLD_BOUNDS, WORLD_SHAPE, WORLD_SLICES


@dataclass(frozen=True)
class Level:
    width: int
    height: int
    people_per_mark: int

    def metadata(self):
        return {
            "width": self.width,
            "height": self.height,
            "peoplePerMark": self.people_per_mark,
        }


ROMANIA = (4_738, 4_810, 24_755, 24_969)


def source_window(source_shape, slice_name, slice_window):
    height, width = source_shape
    if slice_window is None:
        if slice_name == "world":
            return 0, height, 0, width
        if source_shape != WORLD_SHAPE:
            raise ValueError("named slices require the original WorldPop grid")
        slice_window = WORLD_SLICES[slice_name]
    top, bottom, left, right = slice_window
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError("slice indices must define a non-empty window within the source raster")
    return top, bottom, left, right


def levels_for_source(source, tile_width, max_width, pop_target, *, slice_name, slice_window):
    if not 0 < tile_width <= max_width:
        raise ValueError("require 0 < --tile-width <= --max-width")
    if pop_target < 1:
        raise ValueError("--pop-target must be positive")
    with TiffFile(source) as tiff:
        page = tiff.pages[0]
        top, bottom, left, right = source_window((page.imagelength, page.imagewidth), slice_name, slice_window)
    if max_width > right - left:
        raise ValueError(f"--max-width cannot exceed the selected source width of {right - left:,} pixels")

    widths = []
    width = tile_width
    while width < max_width:
        widths.append(width)
        width *= 2
    widths.append(max_width)
    sizes = [(width, max(1, round(Fraction(width * (bottom - top), right - left)))) for width in widths]
    largest_width, largest_height = sizes[-1]
    population_scale = pop_target * largest_width * largest_height
    return [Level(width, height, round(Fraction(population_scale, width * height))) for width, height in sizes]


@dataclass
class Population:
    bounds: tuple[float, float, float, float]
    population: float
    correction: float
    grids: list[tuple[np.ndarray, np.ndarray]]


class Bins:
    def __init__(self, source_shape, target_shape):
        height, width = source_shape
        target_height, target_width = target_shape
        if not (0 < target_height <= height and 0 < target_width <= width):
            raise ValueError("display grid must fit inside the source grid")
        self.rows = np.arange(height) * target_height // height
        self.columns = np.arange(width) * target_width // width
        self.counts = np.zeros(target_shape, dtype=np.float64)
        self.land = np.zeros(target_shape, dtype=bool)

    def add(self, values, valid, top, left):
        height, width = values.shape
        rows = self.rows[top:top + height]
        columns = self.columns[left:left + width]
        local_width = int(columns[-1] - columns[0] + 1)
        local_height = int(rows[-1] - rows[0] + 1)
        size = local_width * local_height
        pixel_rows, pixel_columns = valid.nonzero()
        bins = ((rows[pixel_rows] - rows[0]) * local_width
                + columns[pixel_columns] - columns[0])
        offsets = np.s_[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
        counts = np.bincount(bins, weights=values[valid], minlength=size)
        self.counts[offsets] += counts.reshape(local_height, local_width)
        self.land[offsets] |= np.bincount(bins, minlength=size).reshape(local_height, local_width) > 0


def raster_bounds(page):
    tags = page.geotiff_tags or {}
    if tags.get("GeographicTypeGeoKey") != 4326 or tags.get("GTRasterTypeGeoKey") != 1:
        raise ValueError("expected a WGS 84 GeoTIFF with pixel-area coordinates")
    scale = tags.get("ModelPixelScale", [])
    tiepoint = tags.get("ModelTiepoint", [])
    if len(scale) != 3 or len(tiepoint) != 6 or page.tags.get("ModelTransformationTag"):
        raise ValueError("expected a north-up raster with one origin and a pixel scale")
    scale_x, scale_y, _ = scale
    row_x, row_y, _, longitude, latitude, _ = tiepoint
    if min(scale_x, scale_y) <= 0 or not all(math.isfinite(value) for value in (*scale, *tiepoint)):
        raise ValueError("invalid GeoTIFF scale or origin")
    west = longitude - row_x * scale_x
    north = latitude + row_y * scale_y
    east = west + page.imagewidth * scale_x
    south = north - page.imagelength * scale_y
    if south < -90 or north > 90 or east - west > 360.000001:
        raise ValueError("source raster bounds are outside a geographic grid")
    return west, south, east, north


def correct_romania(values, valid, top, left):
    first_row, last_row, first_column, last_column = ROMANIA
    height, width = values.shape
    start_y = max(first_row, top) - top
    end_y = min(last_row, top + height) - top
    start_x = max(first_column, left) - left
    end_x = min(last_column, left + width) - left
    if start_y >= end_y or start_x >= end_x:
        return values, 0.0
    section = np.s_[start_y:end_y, start_x:end_x]
    # Tifffile can decode a tile into a read-only buffer.
    values = values.copy()
    window = values[section]
    land = valid[section]
    difference = 10 * land.sum() - np.sum(window, dtype=np.float64, where=land)
    window[land] = 10.0
    return values, float(difference)


def read_population(source: Path, levels: list[Level], romania_patch=True, *,
                    slice_name="world", slice_window=None):
    """Stream decoded TIFF tiles once and reduce each requested level directly."""
    with TiffFile(source) as tiff:
        page = tiff.pages[0]
        if page.axes != "YX" or "GDAL_NODATA" not in page.tags:
            raise ValueError("expected a single-channel population raster with a NoData tag")
        bounds = raster_bounds(page)
        # The correction uses absolute source indices in the original WorldPop raster.
        apply_correction = romania_patch and page.shape == WORLD_SHAPE and all(
            math.isclose(value, expected, rel_tol=0, abs_tol=1e-6)
            for value, expected in zip(bounds, WORLD_BOUNDS)
        )
        first_row, last_row, first_column, last_column = source_window(
            (page.imagelength, page.imagewidth), slice_name, slice_window
        )
        if slice_window is not None or slice_name != "world":
            west, south, east, north = bounds
            scale_x = (east - west) / page.imagewidth
            scale_y = (north - south) / page.imagelength
            bounds = (west + first_column * scale_x, north - last_row * scale_y,
                      west + last_column * scale_x, north - first_row * scale_y)
        source_shape = (last_row - first_row, last_column - first_column)
        accumulators = [Bins(source_shape, (level.height, level.width)) for level in levels]
        # Decode at raster precision because tifffile can reject WorldPop's float32 minimum.
        no_data = page.dtype.type(page.tags["GDAL_NODATA"].value.replace(",", "."))
        source_total = 0.0
        correction = 0.0
        for segment, (_, _, top, left, _), _ in page.segments(maxworkers=4):
            if segment is None:
                continue
            # Intersect the TIFF segment with the selected window, excluding edge padding.
            start_y, end_y = max(top, first_row), min(top + segment.shape[1], last_row)
            start_x, end_x = max(left, first_column), min(left + segment.shape[2], last_column)
            if start_y >= end_y or start_x >= end_x:
                continue
            values = segment[0, start_y - top:end_y - top, start_x - left:end_x - left, 0]
            valid = np.isfinite(values) & (values != no_data)
            if np.any(valid & (values < 0)):
                raise ValueError(f"negative population outside NoData at tile ({top}, {left})")
            if not valid.any():
                continue
            source_total += float(np.sum(values, dtype=np.float64, where=valid))
            if apply_correction:
                values, delta = correct_romania(values, valid, start_y, start_x)
                correction += delta
            for bins in accumulators:
                bins.add(values, valid, start_y - first_row, start_x - first_column)

    expected = source_total + correction
    for level, bins in zip(levels, accumulators):
        if not math.isclose(bins.counts.sum(), expected, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(f"the {level.width:,}-pixel map lost population during aggregation")
    return Population(bounds, source_total, correction,
                      [(bins.counts, bins.land) for bins in accumulators])
