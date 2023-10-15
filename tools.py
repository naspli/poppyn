import numpy as np
from numba import njit


@njit
def njit_reduce_resolution(out, arr, factor):
    N = arr.shape[0]
    M = arr.shape[1]
    for i in range(N):
        for j in range(M):
            ii = i // factor
            jj = j // factor
            out[ii, jj] = out[ii, jj] + arr[i, j]


def reduce_resolution(arr, factor):
    arr_reduce = np.zeros([n // factor for n in arr.shape])
    njit_reduce_resolution(arr_reduce, arr, factor)
    return arr_reduce
