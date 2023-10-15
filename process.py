import numpy as np
from numba import njit


@njit
def apply_nbhd_gens(arr, nbhood_func, num_gens):
    N = arr.shape[0]
    M = arr.shape[1]

    for g in range(num_gens):
        next_arr = arr[:, :]
        nbhood = np.zeros((3, 3), dtype=np.float32)
        func_out = np.zeros((3, 3), dtype=np.float32)

        for i in range(N):
            ion = -1 if i != 0 else 0
            iop = 2 if i != N-1 else 1

            for j in range(M):
                jon = -1 if j != 0 else 0
                jop = 2 if j != M-1 else 1

                nbhood.fill(np.nan)
                nbhood[1+ion:1+iop, 1+jon:1+jop] = arr[i+ion:i+iop, j+jon:j+jop]

                func_out.fill(0)
                nbhd_diff = nbhood_func(func_out, nbhood)
                next_arr[i+ion:i+iop, j+jon:j+jop] += nbhd_diff[1+ion:1+iop, 1+jon:1+jop]

        arr[:, :] = next_arr
    return arr


@njit
def flatten_1000(out, nbhood):
    if np.isnan(nbhood[1, 1]):
        return out

    if nbhood[1, 1] > 1000:
        is_smaller = (nbhood <= nbhood[1, 1]).astype(np.float32)
        avg = np.nansum(is_smaller * nbhood) / np.nansum(is_smaller)
        out[:, :] = is_smaller * (avg - nbhood) / 8

    out[:, :] *= (~np.isnan(nbhood)).astype(np.float32)
    out[1, 1] = 0
    out[1, 1] = -np.nansum(out)
    return out


@njit
def i_1000(out, nbhood):
    if np.isnan(nbhood[1, 1]):
        return out

    if nbhood[1, 1] <= 992:
        out[:, :] = -1
    elif nbhood[1, 1] >= 1008:
        out[:, :] = 1

    out[:, :] *= (~np.isnan(nbhood)).astype(np.float32)
    out[1, 1] = 0
    out[1, 1] = -np.nansum(out)
    return out
