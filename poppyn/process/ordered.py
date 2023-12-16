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
    for ip in range(num_points):
        out[ordering[0][-1-ip], ordering[1][-1-ip]] = True
    return out


def select_and_flatten_largest_points(arr, target, delay_reordering=100):
    out = np.zeros_like(arr, dtype=bool)
    num_points = int(np.nansum(arr) // target) + 1
    ordering = None
    for ip in range(num_points):
        if ordering is None or ip % delay_reordering == 0:
            ordering = get_coordinate_ordering(arr)

        idx = (ordering[0][-1-ip], ordering[1][-1-ip])
        out[*idx] = True

        split_remainder = (arr[*idx] - target) / 8
        for ii in range(-1, 2):
            for jj in range(-1, 2):
                arr[idx[0]+ii, idx[1]+jj] += split_remainder
        ip += 1
    return out
