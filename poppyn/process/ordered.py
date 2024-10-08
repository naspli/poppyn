import numpy as np
from tqdm import tqdm

from .neighbours import apply_nbhood_gens, growing_nb_wrapper, nb_flatten2, nb_slide


def get_coordinate_ordering(arr):
    arr = np.nan_to_num(arr)
    ordering = np.argsort(arr, axis=None)
    return np.unravel_index(ordering, arr.shape)


def map_na(na_arr, val_arr):
    return np.where(np.isnan(na_arr), na_arr, val_arr)


def rank_points(arr):
    out = np.zeros_like(arr, dtype=int)
    out = map_na(arr, out)
    ordering = get_coordinate_ordering(arr)
    for i in range(ordering[0].size):
        out[ordering[0][i], ordering[1][i]] = i
    return out


def select_largest_points(arr, target):
    out = np.zeros_like(arr, dtype=bool)
    out = map_na(arr, out)
    ordering = get_coordinate_ordering(arr)
    num_points = int(np.nansum(arr) // target) + 1
    for ip in range(num_points):
        out[ordering[0][-1-ip], ordering[1][-1-ip]] = True
    return out


def select_and_flatten_largest_points(arr, target, no_reorder_before=5, reorder_after=100, progress_bar=False):
    out = np.zeros_like(arr, dtype=bool)
    out = map_na(arr, out)
    arr = arr.copy()
    num_points = int(np.nansum(arr) // target) + 1
    ordering = get_coordinate_ordering(arr)
    ip = 0
    iterator = range(num_points)
    if progress_bar:
        iterator = tqdm(iterator)
    for _ in iterator:
        idx = (ordering[0][-1-ip], ordering[1][-1-ip])
        if ip >= no_reorder_before and arr[*idx] < target:
            percent_diff = int(100 * (target - arr[*idx]) / target)
            apply_nbhood_gens(nb_slide, arr, target, 3, min(1+percent_diff, 10))
            ordering = None

        if ordering is None or ip >= reorder_after:
            ordering = get_coordinate_ordering(arr)
            ip = 0

        out[*idx] = True
        growing_nb_wrapper(nb_flatten2, arr, idx[0], idx[1], target, 100, arr)
        arr[*idx] = np.nan
        ip += 1
    return out
