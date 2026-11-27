"""Rasterise geographic land and lake outlines for the map background."""

import math

import numpy as np

from . import lakes
from .lakes import polygon_crossings, read_polygons
from .projection import EQUAL_AREA, equal_area_bounds, equal_area_northing, equal_earth_parallel


LAND_PATH = lakes.LAKES_PATH.with_name("ne_10m_land.geojson.gz")


class Basemap:
    def __init__(self, bounds, projection, earth):
        self.bounds = bounds
        self.projection = projection
        self.earth = earth
        self.land = read_polygons(LAND_PATH)
        self.lakes = read_polygons(lakes.LAKES_PATH)

    def land_mask(self, width, height):
        west, south, east, north = self.bounds
        if self.earth is not None:
            left, bottom, right, top = self.earth.bounds
            latitudes = inverse_latitudes(row_centres(top, bottom, height),
                                          lambda latitude: equal_earth_parallel(latitude)[0])
            _, scale = equal_earth_parallel(latitudes)
            row_left = self.earth.meridian + left / scale
            steps = (right - left) / (width * scale)
            west = max(float(row_left.min()), self.earth.meridian - 180)
            east = min(float((row_left + width * steps).max()), self.earth.meridian + 180)
        else:
            latitudes = row_centres(north, south, height)
            if self.projection == EQUAL_AREA:
                _, bottom, _, top = equal_area_bounds(self.bounds)
                latitudes = inverse_latitudes(row_centres(top, bottom, height), equal_area_northing)
            row_left = np.full(height, west)
            steps = np.full(height, (east - west) / width)

        mask = np.zeros((height, width), dtype=bool)
        paint_polygons(mask, self.land, latitudes, row_left, steps, west, east, True)
        paint_polygons(mask, self.lakes, latitudes, row_left, steps, west, east, False)
        return mask


def row_centres(top, bottom, height):
    return top - (np.arange(height) + 0.5) * ((top - bottom) / height)


def inverse_latitudes(northings, project):
    # Northing increases monotonically in both projections.
    lower = np.full_like(northings, -90.0)
    upper = np.full_like(northings, 90.0)
    for _ in range(32):
        middle = (lower + upper) / 2
        below = project(middle) < northings
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return (lower + upper) / 2


def paint_polygons(mask, layer, latitudes, row_left, steps, west, east, value):
    bounds, polygons = layer
    left, bottom, right, top = bounds.T
    latitude_hit = (bottom <= latitudes[0]) & (top >= latitudes[-1])
    width = mask.shape[1]
    for turn in range(math.floor((west + 180) / 360), math.floor((east + 180) / 360) + 1):
        shift = 360 * turn
        indices = np.flatnonzero(latitude_hit & (left + shift <= east) & (right + shift >= west))
        for index in indices:
            for row, crossings in polygon_crossings(latitudes, *polygons[index]):
                longitudes = np.clip(crossings + shift, west, east)
                columns = np.ceil((longitudes - row_left[row]) / steps[row] - 0.5).astype(np.intp)
                for first, last in np.clip(columns, 0, width).reshape(-1, 2):
                    mask[row, first:last] = value
