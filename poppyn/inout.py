from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
from dask.array.image import imread
from tifffile import imsave

from .statics import WORLD_FILE, WORLD_SLICES, WORLD_SIZE, AREA_FILE

BASE_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"


def get_data_path(fn):
    fn = Path(fn)
    if not fn.is_absolute():
        fn = DEFAULT_DATA_DIR / fn
    return fn


def load_data(fn=WORLD_FILE, min_land_area=50_000, raw=False):
    path = get_data_path(fn)
    print(f"Reading data from [{path}]")
    darr = imread(str(path))
    if raw:
        return darr
    darr = da.where(darr < 0, np.nan, darr)
    if fn == WORLD_FILE and min_land_area > 0:
        area_path = get_data_path(AREA_FILE)
        area = imread(str(area_path))
        darr = da.where(da.isnan(area), np.nan, darr)
        darr = da.where(area < min_land_area, np.nan, darr)
    return darr.compute()[0]


def save_data(fn, arr):
    path = get_data_path(fn)
    print(f"Dumping data to [{path}]")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    imsave(str(path), arr)


def get_slice_filename(slice_name):
    if not slice_name:
        return WORLD_FILE
    return f"cache/sliced/{slice_name.replace(' ', '')}_only.tif"


def generate_slice(slice_name, arr=None, idx=None):
    print("Generating slice from world file")
    if arr is None:
        arr = load_data(WORLD_FILE)
    if idx is None:
        y, x = WORLD_SLICES[slice_name]
    else:
        y = idx[0], idx[1]
        x = idx[2], idx[3]
    arr_partial = arr[y[0]:y[1], x[0]:x[1]]
    fn = get_slice_filename(slice_name)
    save_data(fn, arr_partial)
    return arr_partial


def get_slice_size(slice_name):
    if not slice_name:
        return WORLD_SIZE
    y, x = WORLD_SLICES[slice_name]
    return y[1]-y[0], x[1]-x[0]


def get_scale(arr, slice_name):
    og_shape = get_slice_size(slice_name)
    return (og_shape[0] / arr.shape[0]) * (og_shape[1] / arr.shape[1])


def load_or_generate_slice(slice_name, idx=None):
    print(f"Fetching slice {slice_name}")
    fn = get_slice_filename(slice_name)
    if get_data_path(fn).exists():
        return load_data(fn)
    else:
        return generate_slice(slice_name, idx=idx)


def get_cache_filename(cache_key):
    return f"cache/processed/poppyn_{cache_key}.tif"


def get_image_filename(cache_key):
    return f"output/poppyn_{cache_key}.png"


def save_image(cache_key, fig):
    fn = get_image_filename(cache_key)
    path = get_data_path(fn)
    print(f"Dumping image to {path}")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    fig.savefig(path, dpi=100)
