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
GEOGRAPHIC = "geographic"
EQUAL_AREA = "equal-area"
EQUAL_AREA_CRS = "EPSG:6933"

# WGS 84 ellipsoid and the 30° standard parallel of EPSG:6933.
_SEMI_MAJOR_METRES = 6_378_137
_FLATTENING = 1 / 298.257223563
_ECCENTRICITY_SQUARED = _FLATTENING * (2 - _FLATTENING)
_ECCENTRICITY = math.sqrt(_ECCENTRICITY_SQUARED)
_SCALE = math.cos(math.radians(30)) / math.sqrt(1 - _ECCENTRICITY_SQUARED / 4)


def equal_area_northing(latitude):
    sine = np.sin(np.deg2rad(latitude))
    q = (1 - _ECCENTRICITY_SQUARED) * (
        sine / (1 - _ECCENTRICITY_SQUARED * sine**2)
        + np.arctanh(_ECCENTRICITY * sine) / _ECCENTRICITY
    )
    return _SEMI_MAJOR_METRES * q / (2 * _SCALE)


def equal_area_bounds(bounds):
    west, south, east, north = bounds
    # Preserve the source window's longitude interval, including its original seam.
    metres_per_degree = _SEMI_MAJOR_METRES * _SCALE * math.pi / 180
    return (west * metres_per_degree, float(equal_area_northing(south)),
            east * metres_per_degree, float(equal_area_northing(north)))


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


def levels_for_source(source, max_width, pop_target, *, tile_width=None, slice_name, slice_window,
                      projection=GEOGRAPHIC):
    if max_width < 1 or (tile_width is not None and tile_width < 1):
        raise ValueError("--width and --tile-width must be positive")
    if pop_target < 1:
        raise ValueError("--pop-target must be positive")
    with TiffFile(source) as tiff:
        page = tiff.pages[0]
        window = source_window((page.imagelength, page.imagewidth), slice_name, slice_window)
        top, bottom, left, right = window
        ratio = Fraction(bottom - top, right - left)
        if projection == EQUAL_AREA:
            west, south, east, north = equal_area_bounds(window_bounds(page, raster_bounds(page), window))
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


class AxisBins:
    """Precompute the positive overlaps between two grids along one axis."""

    def __init__(self, source_size, target_size, source_edges=None):
        if source_edges is None:
            # Use integer units so aligned boundaries are identical.
            source_edges = np.arange(source_size + 1) * target_size
            target_edges = np.arange(target_size + 1) * source_size
        else:
            target_edges = np.linspace(source_edges[0], source_edges[-1], target_size + 1)

        # Each interval between successive edges lies in one source and one output cell.
        edges = np.union1d(source_edges, target_edges)
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


def read_population(source: Path, levels: list[Level], romania_patch=True, *,
                    slice_name="world", slice_window=None, projection=GEOGRAPHIC):
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
        window = source_window((page.imagelength, page.imagewidth), slice_name, slice_window)
        first_row, last_row, first_column, last_column = window
        if slice_window is not None or slice_name != "world":
            bounds = window_bounds(page, bounds, window)
        source_shape = (last_row - first_row, last_column - first_column)
        row_edges = None
        if projection == EQUAL_AREA:
            latitudes = np.linspace(bounds[3], bounds[1], source_shape[0] + 1)
            row_edges = -equal_area_northing(latitudes)  # Rows run from north to south.
        accumulators = [Bins(source_shape, (level.height, level.width), row_edges=row_edges) for level in levels]
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
