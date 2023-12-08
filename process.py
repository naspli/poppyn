import numpy as np
from numba import njit


@njit
def apply_nbhd_gens(arr, nbhood_func, nb_size, target, num_gens):
    N = arr.shape[0]
    M = arr.shape[1]

    middle = (nb_size - 1) // 2

    for g in range(num_gens):
        gen_diff = np.zeros((N, M), dtype=np.float32)
        nbhood = np.zeros((nb_size, nb_size), dtype=np.float32)
        func_out = np.zeros((nb_size, nb_size), dtype=np.float32)

        for i in range(N):
            ion = -middle + np.maximum(middle - i, 0)
            iop = middle + 1 + np.minimum(N - i - 1 - middle, 0)

            for j in range(M):
                jon = -middle + np.maximum(middle - j, 0)
                jop = middle + 1 + np.minimum(M - j - 1 - middle, 0)

                nbhood.fill(np.nan)
                nbhood[middle+ion:middle+iop, middle+jon:middle+jop] = arr[i+ion:i+iop, j+jon:j+jop]

                func_out.fill(0)
                nbhd_diff = nbhood_func(func_out, nbhood, target)
                gen_diff[i+ion:i+iop, j+jon:j+jop] += nbhd_diff[middle+ion:middle+iop, middle+jon:middle+jop]

        arr[:, :] += gen_diff
    return arr


@njit
def nb_flatten(out, nbhood, target):
    nb_size = out.shape[0]
    middle = (nb_size - 1) // 2
    cval = nbhood[middle, middle]

    if np.isnan(cval):
        return out

    is_smaller = (nbhood < cval).astype(np.float32)
    num_smaller = np.nansum(is_smaller)
    if num_smaller == 0:
        return out

    if cval > target:
        out[:, :] += is_smaller * (cval - target) / (out.size - 1)
    elif cval < target:
        out[:, :] += -is_smaller * nbhood / (out.size - 1)

    out[:, :] *= (~np.isnan(nbhood)).astype(np.float32)
    out[middle, middle] = 0
    out[middle, middle] = -np.nansum(out)
    return out


@njit
def nb_trickle(out, nbhood, target):
    nb_size = out.shape[0]
    middle = (nb_size - 1) // 2
    cval = nbhood[middle, middle]

    if np.isnan(cval):
        return out

    if cval <= target - (out.size - 1):
        out[:, :] = -1
    elif cval >= target + (out.size - 1):
        out[:, :] = 1

    out[:, :] *= (~np.isnan(nbhood)).astype(np.float32)
    out[middle, middle] = 0
    out[middle, middle] = -np.nansum(out)
    return out
