from pathlib import Path

import numpy as np
import dask.array as da
from dask.array.image import imread
from tifffile import imsave

from statics import WORLD_FILE, WORLD_SLICES

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def get_data_path(fn):
    fn = Path(fn)
    if not fn.is_absolute():
        fn = DEFAULT_DATA_DIR / fn
    return fn


def load_data(fn=WORLD_FILE, raw=False):
    fn = get_data_path(fn)
    darr = imread(str(fn))
    if raw:
        return darr
    darr = da.where(darr < 0, np.nan, darr)
    return darr.compute()[0]


def save_data(fn, arr):
    fn = get_data_path(fn)
    imsave(str(fn), arr)


def get_slice_filename(key):
    return WORLD_FILE.split(".")[0] + f"_{key}_only.tif"


def generate_slice(key, arr=None):
    if arr is None:
        arr = load_data(WORLD_FILE)
    y, x = WORLD_SLICES[key]
    arr_partial = arr[y[0]:y[1], x[0]:x[1]]
    save_data(get_slice_filename(key), arr_partial)
    return arr_partial


def load_slice(key):
    fn = get_slice_filename(key)
    if get_data_path(fn).exists():
        return load_data(fn)
    else:
        return generate_slice(key)
