"""Identify open lake water on the source grid using Natural Earth outlines."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


LAKES_PATH = Path(__file__).with_name("data") / "ne_10m_lakes.geojson.gz"
SHORELINE_BAND = 2


class LakeMask:
    def __init__(self, bounds, shape):
        self.west, south, east, self.north = bounds
        height, width = shape
        self.dx = (east - self.west) / width
        self.dy = (self.north - south) / height
        self.bounds, self.polygons = read_polygons(LAKES_PATH)

    def interior(self, top, left, shape):
        """Return an open-water mask, reading the shoreline band beyond crop and tile edges."""
        height, width = shape
        band = SHORELINE_BAND
        longitudes = self.west + (np.arange(left - band, left + width + band) + 0.5) * self.dx
        latitudes = self.north - (np.arange(top - band, top + height + band) + 0.5) * self.dy
        west, south, east, north = self.bounds.T
        latitudes_overlap = (south <= latitudes[0]) & (north >= latitudes[-1])
        water = None
        # The lake polygons use longitudes in [-180, 180]. A padded chunk can cross that seam.
        for turn in range(math.floor((longitudes[0] + 180) / 360), math.floor((longitudes[-1] + 180) / 360) + 1):
            xs = longitudes - 360 * turn
            indices = np.flatnonzero(latitudes_overlap & (west <= xs[-1]) & (east >= xs[0]))
            if not indices.size:
                continue
            if water is None:
                water = np.zeros((latitudes.size, longitudes.size), dtype=bool)
            for index in indices:
                fill_polygon(water, xs, latitudes, *self.polygons[index])
        if water is None:
            return None
        # Keep a cell as open water only when its entire 5-by-5 neighbourhood is water.
        size = 2 * band + 1
        water = sliding_window_view(water, size, axis=1).all(axis=-1)
        return sliding_window_view(water, size, axis=0).all(axis=-1)


def read_polygons(path):
    """Read the geographic bounds and paired edge endpoints for each polygon."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        features = json.load(source)["features"]
    bounds, polygons = [], []
    for feature in features:
        geometry = feature["geometry"]
        parts = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            parts = [parts]
        for polygon in parts:
            rings = [np.asarray(ring) for ring in polygon]
            vertices = np.concatenate(rings)
            bounds.append([*vertices.min(axis=0), *vertices.max(axis=0)])
            polygons.append((np.concatenate([ring[:-1] for ring in rings]),
                             np.concatenate([ring[1:] for ring in rings])))
    return np.asarray(bounds).reshape(-1, 4), polygons


def polygon_crossings(latitudes, starts, ends):
    """Yield each intersected row and its sorted longitude crossings, including holes."""
    x0, y0 = starts.T
    x1, y1 = ends.T
    low, high = np.minimum(y0, y1), np.maximum(y0, y1)
    for row in np.flatnonzero((low.min() <= latitudes) & (latitudes < high.max())):
        y = latitudes[row]
        crosses = (low <= y) & (y < high)
        x = x0[crosses] + (y - y0[crosses]) * (x1[crosses] - x0[crosses]) / (y1[crosses] - y0[crosses])
        yield row, np.sort(x)


def fill_polygon(water, longitudes, latitudes, starts, ends):
    for row, crossings in polygon_crossings(latitudes, starts, ends):
        for start, end in np.searchsorted(longitudes, crossings).reshape(-1, 2):
            water[row, start:end] = True
