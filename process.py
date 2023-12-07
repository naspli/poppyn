import numpy as np
from numba import njit


@njit
def apply_nbhd_gens(arr, nbhood_func, target, num_gens):
    N = arr.shape[0]
    M = arr.shape[1]

    for g in range(num_gens):
        gen_diff = np.zeros((N, M), dtype=np.float32)
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
                nbhd_diff = nbhood_func(func_out, nbhood, target)
                gen_diff[i+ion:i+iop, j+jon:j+jop] += nbhd_diff[1+ion:1+iop, 1+jon:1+jop]

        arr[:, :] += gen_diff
    return arr


@njit
def nb_flatten(out, nbhood, target):
    if np.isnan(nbhood[1, 1]):
        return out

    is_smaller = (nbhood < nbhood[1, 1]).astype(np.float32)
    num_smaller = np.nansum(is_smaller)
    if num_smaller > 0:
        # out_min = 0
        # out_max = 0
        if nbhood[1, 1] > target:
            out[:, :] += is_smaller * (nbhood[1, 1] - target) / 8
            # out_max = (nbhood[1, 1] - target) / 8
        elif nbhood[1, 1] < target:
            out[:, :] += -is_smaller * nbhood / 8
        #     out_min = (nbhood[1, 1] - target) / 8
        # out[:, :] += (out < out_min).astype(np.float32) * (out_min - out)
        # out[:, :] += (out > out_max).astype(np.float32) * (out_max - out)

    out[:, :] *= (~np.isnan(nbhood)).astype(np.float32)
    out[1, 1] = 0
    out[1, 1] = -np.nansum(out)
    return out


@njit
def nb_trickle(out, nbhood):
    if np.isnan(nbhood[1, 1]):
        return out

    if nbhood[1, 1] <= target - 8:
        out[:, :] = -1
    elif nbhood[1, 1] >= target + 8:
        out[:, :] = 1

    out[:, :] *= (~np.isnan(nbhood)).astype(np.float32)
    out[1, 1] = 0
    out[1, 1] = -np.nansum(out)
    return out
