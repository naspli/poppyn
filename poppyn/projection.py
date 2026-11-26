"""Projection coordinates in metres on the WGS 84 ellipsoid."""

import math

import numpy as np


GEOGRAPHIC = "geographic"
EQUAL_AREA = "equal-area"
EQUAL_AREA_CRS = "EPSG:6933"
EQUAL_EARTH = "equal-earth"

_SEMI_MAJOR_METRES = 6_378_137
_FLATTENING = 1 / 298.257223563
_ECCENTRICITY_SQUARED = _FLATTENING * (2 - _FLATTENING)
_ECCENTRICITY = math.sqrt(_ECCENTRICITY_SQUARED)


def authalic_q(latitude):
    """Ellipsoid area between the equator and a parallel, per radian of longitude and a²/2."""
    sine = np.sin(np.deg2rad(latitude))
    return (1 - _ECCENTRICITY_SQUARED) * (
        sine / (1 - _ECCENTRICITY_SQUARED * sine**2)
        + np.arctanh(_ECCENTRICITY * sine) / _ECCENTRICITY
    )


_CYLINDRICAL_SCALE = math.cos(math.radians(30)) / math.sqrt(1 - _ECCENTRICITY_SQUARED / 4)


def equal_area_northing(latitude):
    return _SEMI_MAJOR_METRES * authalic_q(latitude) / (2 * _CYLINDRICAL_SCALE)


def equal_area_bounds(bounds):
    west, south, east, north = bounds
    metres_per_degree = _SEMI_MAJOR_METRES * _CYLINDRICAL_SCALE * math.pi / 180
    return (west * metres_per_degree, float(equal_area_northing(south)),
            east * metres_per_degree, float(equal_area_northing(north)))


# Šavrič, Patterson & Jenny, https://doi.org/10.1080/13658816.2018.1504949.
_A1, _A2, _A3, _A4 = 1.340264, -0.081106, 0.000893, 0.003796
_SIN_60 = math.sqrt(3) / 2
_POLAR_Q = float(authalic_q(90))
_AUTHALIC_RADIUS = _SEMI_MAJOR_METRES * math.sqrt(_POLAR_Q / 2)


def _equal_earth_y(theta):
    square = theta**2
    return _AUTHALIC_RADIUS * theta * (_A1 + _A2 * square + square**3 * (_A3 + _A4 * square))


def _equal_earth_dy(theta):
    square = theta**2
    return _AUTHALIC_RADIUS * (_A1 + 3 * _A2 * square + square**3 * (7 * _A3 + 9 * _A4 * square))


def equal_earth_parallel(latitude):
    """Return northing and metres per degree of longitude along the parallel."""
    theta = np.arcsin(_SIN_60 * authalic_q(latitude) / _POLAR_Q)
    return _equal_earth_y(theta), _equal_earth_scale(theta)


def _equal_earth_scale(theta):
    return _AUTHALIC_RADIUS**2 * (math.pi / 180) * np.cos(theta) / (_SIN_60 * _equal_earth_dy(theta))


def equal_earth_bands(latitude_edges):
    """Return southward edges, mean horizontal scales and area weights for two bands per row."""
    q = authalic_q(latitude_edges)
    edges = -_equal_earth_y(np.arcsin(_SIN_60 * q / _POLAR_Q))
    area_weights = q[:-1] - q[1:]
    # Width × height equals the exact area per degree of longitude in each band.
    scale = (_SEMI_MAJOR_METRES**2 * math.pi / 360) * area_weights / np.diff(edges)
    row_weights = area_weights.reshape(-1, 2)
    return edges, scale, (row_weights / row_weights.sum(axis=1, keepdims=True)).ravel()


class EqualEarth:
    def __init__(self, geographic_bounds, *, world=False):
        west, south, east, north = geographic_bounds
        # Retain the raster seam for a world map. Centre regional maps on their slice.
        self.meridian = (west + east) / 2
        globe = world and math.isclose(east - west, 360, abs_tol=0.0001)
        if globe:
            south, north = -90, 90
        near_equator = max(south, 0) if north >= 0 else north
        _, widest_scale = equal_earth_parallel(near_equator)
        y_south, scale_south = equal_earth_parallel(south)
        y_north, scale_north = equal_earth_parallel(north)
        half_width = 180 if globe else (east - west) / 2
        half_width = float(half_width * widest_scale)
        self.bounds = (-half_width, float(y_south), half_width, float(y_north))
        self.max_longitude_delta = min(180, half_width / min(float(scale_south), float(scale_north)))

    def metadata(self):
        return {
            "crs": f"+proj=eqearth +lon_0={self.meridian:.16g} +datum=WGS84 +units=m +type=crs",
            "centralMeridian": self.meridian,
            "projectedBounds": self.bounds,
        }

    def outside(self, width, height):
        west, south, east, north = self.bounds
        ys = np.linspace(north, south, height + 1)
        xs = np.linspace(west, east, width + 1)
        # Within each output row, choose the widest point of the curved outline.
        nearest_y = np.maximum(ys[1:], 0) + np.minimum(ys[:-1], 0)
        theta = nearest_y / (_AUTHALIC_RADIUS * _A1)
        for _ in range(5):
            theta -= (_equal_earth_y(theta) - nearest_y) / _equal_earth_dy(theta)
        half_width = 180 * _equal_earth_scale(theta)
        nearest_x = np.maximum(xs[:-1], 0) - np.minimum(xs[1:], 0)
        return nearest_x[None, :] >= half_width[:, None]
