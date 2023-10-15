# https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif
# Estimated total number of people per grid-cell. The dataset is available to download in Geotiff format at a resolution
# of 30 arc (approximately 1km at the equator). The projection is Geographic Coordinate System, WGS84. The units are
# number of people per pixel. The mapping approach is Random Forest-based dasymetric redistribution.
WORLD_FILE = "ppp_2020_1km_Aggregated.tif"
WORLD_SIZE = (18720, 43200)

WORLD_SLICES = {
    "GB": ((2500, 4500), (20000, 22000)),
    "London": ((3830, 3960), (21480, 21690))
}