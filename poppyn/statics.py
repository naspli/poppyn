# https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif
# Estimated total number of people per grid-cell. The dataset is available to download in Geotiff format at a resolution
# of 30 arc (approximately 1km at the equator). The projection is Geographic Coordinate System, WGS84. The units are
# number of people per pixel. The mapping approach is Random Forest-based dasymetric redistribution.
WORLD_FILE = "raw/ppp_2020_1km_Aggregated.tif"
# https://data.worldpop.org/GIS/Pixel_area/Global_2000_2020/0_Mosaicked/global_px_area_1km.tif
# similarly land-area from WorldPop
AREA_FILE = "raw/global_px_area_1km.tif"
WORLD_SIZE = (18720, 43200)

WORLD_SLICES = {
    # Large
    "Europe": ((1750, 6250), (19750, 27750)),
    "US": ((18720-15250, 18720-10250), (6500, 13500)),
    "India": ((18720-13100, 18720-9100), (29450, 32950)),
    "Far East": ((18720-14250, 18720-10750), (32950, 38950)),
    
    # Medium
    "Britain": ((2500, 4500), (20000, 22000)),
    "US East Coast": ((4500, 6500), (11500, 13500)),
    "Japan": ((4500, 6500), (37100, 39100)),

    # Small
    "London": ((3800, 4000), (21490, 21690)),
}

BAD_DATA_SLICES = [
    (((4738, 4810), (24755, 24969)), 10)  # weird issue in dataset in south of Romania
]
