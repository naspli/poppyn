"""Read WorldPop counts into aligned, population-preserving display grids."""

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path

import numpy as np
from tifffile import TiffFile

from .projection import (
    EQUAL_AREA, EQUAL_EARTH, GEOGRAPHIC, EqualEarth, equal_area_bounds, equal_area_northing, equal_earth_bands,
)
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


def source_window(source_shape, bounds, slice_name, slice_window):
    height, width = source_shape
    if slice_window is None:
        if slice_name == "world":
            return 0, height, 0, width
        west, south, east, north = WORLD_SLICES[slice_name]
        raster_west, raster_south, raster_east, raster_north = bounds
        cols_per_degree = width / (raster_east - raster_west)
        rows_per_degree = height / (raster_north - raster_south)
        # Round subpixel noise from the stored coordinates before snapping outwards.
        left = max(0, math.floor(round((west - raster_west) * cols_per_degree, 3)))
        right = min(width, math.ceil(round((east - raster_west) * cols_per_degree, 3)))
        top = max(0, math.floor(round((raster_north - north) * rows_per_degree, 3)))
        bottom = min(height, math.ceil(round((raster_north - south) * rows_per_degree, 3)))
        if not (top < bottom and left < right):
            raise ValueError(f"named slice {slice_name!r} does not overlap the source raster")
        return top, bottom, left, right
    top, bottom, left, right = slice_window
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError("slice indices must define a non-empty window within the source raster")
    return top, bottom, left, right


def levels_for_source(source, max_width, pop_target, *, tile_width=None, slice_name, slice_window,
                      projection=GEOGRAPHIC):
    if max_width < 1 or (tile_width is not None and tile_width < 1):
        raise ValueError("--width and --tile-width must be positive")
    if pop_target < 1:
        raise ValueError("--pop-target must be positive")
    with TiffFile(source) as tiff:
        page = tiff.pages[0]
        bounds = raster_bounds(page)
        window = source_window(page.shape, bounds, slice_name, slice_window)
        top, bottom, left, right = window
        ratio = Fraction(bottom - top, right - left)
        if projection != GEOGRAPHIC:
            bounds = window_bounds(page, bounds, window)
            projected = (EqualEarth(bounds, world=slice_name == "world").bounds if projection == EQUAL_EARTH
                         else equal_area_bounds(bounds))
            west, south, east, north = projected
            ratio = (north - south) / (east - west)
    if max_width > right - left:
        raise ValueError(f"--width cannot exceed the selected source width of {right - left:,} pixels")

    widths = []
    width = tile_width or max_width
    while width < max_width:
        widths.append(width)
        width *= 2
    widths.append(max_width)
    sizes = [(width, max(1, round(width * ratio))) for width in widths]
    largest_width, largest_height = sizes[-1]
    population_scale = pop_target * largest_width * largest_height
    return [Level(width, height, round(Fraction(population_scale, width * height))) for width, height in sizes]


@dataclass
class Population:
    bounds: tuple[float, float, float, float]
    population: float
    correction: float
    grids: list[tuple[np.ndarray, np.ndarray]]
    window: tuple[int, int, int, int]


class AxisBins:
    """Precompute the positive overlaps between two grids along one axis."""

    def __init__(self, source_size, target_size, source_edges=None, *, target_bounds=None):
        if source_edges is None:
            # Use integer units so aligned boundaries are identical.
            source_edges = np.arange(source_size + 1) * target_size
            target_edges = np.arange(target_size + 1) * source_size
        else:
            bounds = target_bounds or (source_edges[0], source_edges[-1])
            target_edges = np.linspace(*bounds, target_size + 1)

        # Each interval between successive edges lies in one source and one output cell.
        edges = np.union1d(source_edges, target_edges)
        if target_bounds is not None:
            edges = edges[(edges >= source_edges[0]) & (edges <= source_edges[-1])]
        self.source = np.searchsorted(source_edges, edges[:-1], side="right") - 1
        self.target = np.searchsorted(target_edges, edges[:-1], side="right") - 1
        self.weights = np.diff(edges) / np.diff(source_edges)[self.source]
        self.offsets = np.searchsorted(self.source, np.arange(source_size + 1))

    def sum(self, counts, land, start):
        chunk = slice(self.offsets[start], self.offsets[start + counts.shape[0]])
        source = self.source[chunk] - start
        target = self.target[chunk]
        starts = np.r_[0, np.flatnonzero(target[1:] != target[:-1]) + 1]
        totals = np.add.reduceat(counts[source] * self.weights[chunk, None], starts, axis=0)
        covered = np.logical_or.reduceat(land[source], starts, axis=0)
        return totals, covered, slice(target[0], target[-1] + 1)


class Bins:
    def __init__(self, source_shape, target_shape, *, row_edges=None):
        height, width = source_shape
        target_height, target_width = target_shape
        if not (target_height > 0 and 0 < target_width <= width and (row_edges is not None or target_height <= height)):
            raise ValueError("display grid must fit inside the source grid")
        self.rows = AxisBins(height, target_height, row_edges)
        self.columns = AxisBins(width, target_width)
        self.counts = np.zeros(target_shape, dtype=np.float64)
        self.land = np.zeros(target_shape, dtype=bool)

    def add(self, values, valid, top, left):
        counts = np.zeros(values.shape, dtype=np.float64)
        np.copyto(counts, values, where=valid)

        # Separable weights give each output cell its share of the overlap area.
        counts, land, rows = self.rows.sum(counts, valid, top)
        counts, land, columns = self.columns.sum(counts.T, land.T, left)
        self.counts[rows, columns] += counts.T
        self.land[rows, columns] |= land.T


class EqualEarthChunk:
    """Represent a TIFF chunk as horizontal bands with row-specific longitudinal scale."""

    def __init__(self, values, valid, first_band, scale, weight, west, step):
        height, self.width = values.shape
        self.first_band = first_band
        self.band_rows = np.repeat(np.arange(height), 2)[:, None]
        self.scale = scale[:, None]
        self.weight = weight[:, None]
        self.west, self.step = west, step
        self.land_prefix = np.pad(np.cumsum(valid, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
        self.set_population(values, valid)

    def set_population(self, values, valid):
        self.counts = np.where(valid, values, 0)
        self.prefix = np.pad(np.cumsum(self.counts, axis=1, dtype=np.float64), ((0, 0), (1, 0)))

    def positions(self, eastings):
        degrees = np.clip(eastings / self.scale, -180, 180)
        return np.clip((degrees - self.west) / self.step, 0, self.width)

    def cumulative(self, positions):
        columns = np.minimum(positions.astype(np.intp), self.width - 1)
        counts = self.prefix[self.band_rows, columns] + (positions - columns) * self.counts[self.band_rows, columns]
        return counts * self.weight

    def total(self, west, east):
        positions = self.positions([west, east])
        return float(np.diff(self.cumulative(positions), axis=1).sum())

    def horizontal(self, edges):
        left, right = np.clip([self.west, self.west + self.width * self.step], -180, 180)
        left = np.min(left * self.scale)
        right = np.max(right * self.scale)
        first = max(0, np.searchsorted(edges, left, side="right") - 1)
        last = min(len(edges) - 1, np.searchsorted(edges, right))
        if first >= last:
            return None
        positions = self.positions(edges[first:last + 1])
        counts = np.diff(self.cumulative(positions), axis=1)
        begin, end = positions[:, :-1], positions[:, 1:]
        valid_before = self.land_prefix[self.band_rows, begin.astype(np.intp)]
        valid_after = self.land_prefix[self.band_rows, np.ceil(end).astype(np.intp)]
        land = (end > begin) & (valid_after > valid_before)
        return counts, land, slice(first, last)


class EqualEarthBins:
    def __init__(self, row_edges, level, bounds):
        west, south, east, north = bounds
        # The output spans the source latitudes. Include any round-off at their ends.
        vertical = (min(-north, row_edges[0]), max(-south, row_edges[-1]))
        self.rows = AxisBins(len(row_edges) - 1, level.height, row_edges, target_bounds=vertical)
        self.columns = np.linspace(west, east, level.width + 1)
        self.counts = np.zeros((level.height, level.width), dtype=np.float64)
        self.land = np.zeros(self.counts.shape, dtype=bool)

    def add(self, chunk):
        horizontal = chunk.horizontal(self.columns)
        if horizontal is None:
            return
        counts, land, columns = horizontal
        counts, land, rows = self.rows.sum(counts, land, chunk.first_band)
        self.counts[rows, columns] += counts
        self.land[rows, columns] |= land


def longitude_windows(left_edge, step, width, earth):
    """Read any part of the source (including across the antimeridian) that can enter the crop."""
    radius = earth.max_longitude_delta
    for shift in (-360, 0, 360):
        offset = left_edge - earth.meridian + shift
        left = max(0, math.floor((-radius - offset) / step))
        right = min(width, math.ceil((radius - offset) / step))
        if left < right:
            yield left, right, shift


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


def window_bounds(page, bounds, window):
    top, bottom, left, right = window
    west, south, east, north = bounds
    scale_x = (east - west) / page.imagewidth
    scale_y = (north - south) / page.imagelength
    return (west + left * scale_x, north - bottom * scale_y,
            west + right * scale_x, north - top * scale_y)


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


def population_chunks(page, first_row, last_row, column_windows):
    # Decode at raster precision because tifffile can reject WorldPop's float32 minimum.
    no_data = page.dtype.type(page.tags["GDAL_NODATA"].value.replace(",", "."))
    for segment, (_, _, top, left, _), _ in page.segments(maxworkers=4):
        if segment is None:
            continue
        start_y, end_y = max(top, first_row), min(top + segment.shape[1], last_row)
        if start_y >= end_y:
            continue
        for first_column, last_column, shift in column_windows:
            start_x, end_x = max(left, first_column), min(left + segment.shape[2], last_column)
            if start_x >= end_x:
                continue
            values = segment[0, start_y - top:end_y - top, start_x - left:end_x - left, 0]
            valid = np.isfinite(values) & (values != no_data)
            if np.any(valid & (values < 0)):
                raise ValueError(f"negative population outside NoData at tile ({top}, {left})")
            if valid.any():
                yield values, valid, start_y, start_x, shift


def read_population(source: Path, levels: list[Level], romania_patch=True, *,
                    slice_name="world", slice_window=None, projection=GEOGRAPHIC):
    """Stream decoded TIFF tiles once and reduce each requested level directly."""
    with TiffFile(source) as tiff:
        page = tiff.pages[0]
        if page.axes != "YX" or "GDAL_NODATA" not in page.tags:
            raise ValueError("expected a single-channel population raster with a NoData tag")
        bounds = raster_bounds(page)
        raster_west, _, raster_east, _ = bounds
        # The correction uses absolute source indices in the original WorldPop raster.
        apply_correction = romania_patch and page.shape == WORLD_SHAPE and all(
            math.isclose(value, expected, rel_tol=0, abs_tol=1e-6)
            for value, expected in zip(bounds, WORLD_BOUNDS)
        )
        window = source_window(page.shape, bounds, slice_name, slice_window)
        first_row, last_row, first_column, last_column = window
        if slice_window is not None or slice_name != "world":
            bounds = window_bounds(page, bounds, window)
        source_shape = (last_row - first_row, last_column - first_column)
        earth = EqualEarth(bounds, world=slice_name == "world") if projection == EQUAL_EARTH else None
        if earth is None:
            row_edges = None
            if projection == EQUAL_AREA:
                latitudes = np.linspace(bounds[3], bounds[1], source_shape[0] + 1)
                row_edges = -equal_area_northing(latitudes)  # Rows run from north to south.
            column_windows = [(first_column, last_column, 0)]
            accumulators = [Bins(source_shape, (level.height, level.width), row_edges=row_edges) for level in levels]
        else:
            latitudes = np.linspace(bounds[3], bounds[1], 2 * source_shape[0] + 1)
            row_edges, scales, weights = equal_earth_bands(latitudes)
            step = (raster_east - raster_west) / page.imagewidth
            column_windows = list(longitude_windows(raster_west, step, page.imagewidth, earth))
            accumulators = [EqualEarthBins(row_edges, level, earth.bounds) for level in levels]
        source_total = 0.0
        correction = 0.0
        for values, valid, top, left, shift in population_chunks(page, first_row, last_row, column_windows):
            corrected, delta = correct_romania(values, valid, top, left) if apply_correction else (values, 0.0)
            if earth is None:
                source_total += float(np.sum(values, dtype=np.float64, where=valid))
                correction += delta
                for bins in accumulators:
                    bins.add(corrected, valid, top - first_row, left - first_column)
            else:
                first_band = 2 * (top - first_row)
                bands = slice(first_band, first_band + 2 * values.shape[0])
                west = raster_west - earth.meridian + left * step + shift
                chunk = EqualEarthChunk(values, valid, first_band, scales[bands], weights[bands], west, step)
                west, _, east, _ = earth.bounds
                included = chunk.total(west, east)
                source_total += included
                if corrected is not values:
                    chunk.set_population(corrected, valid)
                    correction += chunk.total(west, east) - included
                for bins in accumulators:
                    bins.add(chunk)

    expected = source_total + correction
    for level, bins in zip(levels, accumulators):
        if not math.isclose(bins.counts.sum(), expected, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(f"the {level.width:,}-pixel map lost population during aggregation")
    return Population(bounds, source_total, correction,
                      [(bins.counts, bins.land) for bins in accumulators], window)
