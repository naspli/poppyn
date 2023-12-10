import numpy as np
from numba import njit


@njit
def njit_reduce_resolution(out, arr, factor):
    N = arr.shape[0]
    M = arr.shape[1]

    ii = 0
    ic = 0
    for i in range(N):

        jj = 0
        jc = 0
        for j in range(M):
            if np.isnan(out[ii, jj]):
                out[ii, jj] = arr[i, j]
            elif not np.isnan(arr[i, j]):
                out[ii, jj] += arr[i, j]

            jc += 1
            if jc == factor:
                jc = 0
                jj += 1

        ic += 1
        if ic == factor:
            ic = 0
            ii += 1
    return out


def reduce_resolution(arr, factor):
    arr_reduce = np.zeros(tuple([(n // factor) + 1 for n in arr.shape]), dtype=arr.dtype)
    arr_reduce.fill(np.nan)
    return njit_reduce_resolution(arr_reduce, arr, factor)
