"""Geographic bounds for the source raster and named map regions."""

WORLD_SHAPE = (18_720, 43_200)
WORLD_BOUNDS = (-180.001249265, -72.00041617729, 179.998749295, 83.99958319871)

# WGS 84 degrees in (west, south, east, north) order.
WORLD_SLICES = {
    # Existing presets retain their original WorldPop extent.
    "europe": (-15.417917, 31.916250, 51.248750, 69.416250),
    "us": (-125.834583, 13.416250, -67.501250, 55.082917),
    "india": (65.415416, 3.832917, 94.582083, 37.166250),
    "fareast": (94.582083, 17.582917, 144.582083, 46.749583),
    "britain": (-13.334583, 46.499583, 3.332083, 63.166250),
    "useastcoast": (-84.167916, 29.832917, -67.501250, 46.499583),
    "japan": (129.165416, 29.832917, 145.832083, 46.499583),
    "london": (-0.917917, 50.666250, 0.748750, 52.332917),
    "eastasia": (112.965416, 27.982917, 147.048749, 47.016250),
    "africa": (-26, -36, 57, 38),
    "australia": (112, -44, 154, -10),
    "bangladesh-delta": (87, 20, 93, 27.5),
    "brazil": (-74, -34, -34, 6),
    "canada": (-142, 41, -52, 83.5),
    "central-america-caribbean": (-118, 5, -58, 33),
    "china": (73, 17, 135, 54),
    "great-lakes": (-93, 40, -74, 50),
    "iberia": (-10, 35, 5, 44.5),
    "indonesia": (94, -12, 142, 8),
    "java": (105, -9.5, 115, -5.5),
    "kanto": (137.8, 34.4, 141, 37.1),
    "mediterranean": (-7, 28, 38, 47),
    "middle-east": (25, 10, 64, 43),
    "new-zealand": (165, -48, 179.5, -33),
    "nile-delta": (29, 28, 33, 32.5),
    "scandinavia": (3, 54, 33, 72),
    "south-america": (-82, -56, -34, 13.5),
    "southeast-asia": (92, -12, 142, 28),
    "us-west": (-126, 30, -100, 51),
}
