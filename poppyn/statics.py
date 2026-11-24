"""Named windows in the original WorldPop raster."""

WORLD_SHAPE = (18_720, 43_200)
WORLD_BOUNDS = (-180.001249265, -72.00041617729, 179.998749295, 83.99958319871)

# Half-open source row and column ranges in north-up raster order.
WORLD_SLICES = {
    "europe": (1750, 6250, 19750, 27750),
    "us": (3470, 8470, 6500, 13500),
    "india": (5620, 9620, 29450, 32950),
    "fareast": (4470, 7970, 32950, 38950),
    "britain": (2500, 4500, 20000, 22000),
    "useastcoast": (4500, 6500, 11500, 13500),
    "japan": (4500, 6500, 37100, 39100),
    "london": (3800, 4000, 21490, 21690),
    # Approx. 113–147°E and 28–47°N, snapped outward from the original preview.
    "eastasia": (4438, 6722, 35156, 39246),
}
