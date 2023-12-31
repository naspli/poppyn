import numpy as np
from numba import njit


@njit
def njit_reduce_resolution(out, arr, factor, min_pop):
    N = arr.shape[0]
    M = arr.shape[1]

    num_nan = np.zeros_like(out)

    ii = 0
    next_i = int(factor)
    for i in range(N):

        jj = 0
        next_j = int(factor)
        for j in range(M):
            if np.isnan(arr[i, j]):
                num_nan[ii, jj] += 1
            else:
                out[ii, jj] += arr[i, j]

            while j >= next_j:
                jj += 1
                next_j = int(factor * (jj + 1))

        while i >= next_i:
            ii += 1
            next_i = int(factor * (ii+1))

    out = np.where((num_nan > 0.5 * factor ** 2) & (out < min_pop), np.nan, out)
    return out


def reduce_resolution(arr, factor=None, max_size=None, min_pop=1):
    if int(max_size is None) + int(factor is None) != 1:
        raise ValueError("Provide exactly one of factor or max_size")
    if max_size is not None:
        factor = max(arr.shape) / max_size

    if factor < 1.0:
        return arr

    arr_reduce = np.zeros(tuple([int((n / factor) + 0.5) for n in arr.shape]), dtype=arr.dtype)

    print(f"Reducing resolution from {arr.shape} to {arr_reduce.shape}")
    return njit_reduce_resolution(arr_reduce, arr, factor, min_pop)
