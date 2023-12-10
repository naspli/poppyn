import numpy as np
from numba import njit


def get_coordinate_ordering(arr):
    arr = np.nan_to_num(arr)
    ordering = np.argsort(arr, axis=None)
    return np.unravel_index(ordering, arr.shape)


def rank_points(arr):
    out = np.zeros_like(arr, dtype=int)
    ordering = get_coordinate_ordering(arr)
    for i in range(ordering[0].size):
        out[ordering[0][i], ordering[1][i]] = i
    return out


def select_largest_points(arr, target):
    out = np.zeros_like(arr, dtype=bool)
    ordering = get_coordinate_ordering(arr)
    num_points = int(np.nansum(arr) // target) + 1
    S = ordering[0].size
    for i in range(num_points):
        out[ordering[0][S-1-i], ordering[1][S-1-i]] = True
    return out
