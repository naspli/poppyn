# https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif
# Estimated total number of people per grid-cell. The dataset is available to download in Geotiff format at a resolution
# of 30 arc (approximately 1km at the equator). The projection is Geographic Coordinate System, WGS84. The units are
# number of people per pixel. The mapping approach is Random Forest-based dasymetric redistribution.
WORLD_FILE = "ppp_2020_1km_Aggregated.tif"
# https://data.worldpop.org/GIS/Pixel_area/Global_2000_2020/0_Mosaicked/global_px_area_1km.tif
# similarly land-area from WorldPop
AREA_FILE = "global_px_area_1km.tif"
WORLD_SIZE = (18720, 43200)

WORLD_SLICES = {
    "Britain": ((2500, 4500), (20000, 22000)),
    "London": ((3830, 3960), (21480, 21690)),
    "US East Coast": ((4500, 6500), (11500, 13500)),
    "Japan": ((4500, 6500), (37100, 39100))
}
