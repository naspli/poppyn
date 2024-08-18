import numpy as np
from numba import njit


@njit
def nb_slide(nbhood, i, j, target):
    out = np.zeros_like(nbhood)
    if np.isnan(nbhood[i, j]):
        return out

    cval = nbhood[i, j]
    is_smaller = (nbhood < cval).astype(np.float32)
    num_smaller = np.nansum(is_smaller)
    if num_smaller == 0:
        return out

    if cval > target:
        out[:, :] += is_smaller * (cval - target) / (out.size - 1)
    elif cval < target:
        out[:, :] += -is_smaller * nbhood / (out.size - 1)

    out[:, :] *= (~np.isnan(nbhood)).astype(np.float32)
    out[i, j] = 0
    out[i, j] = -np.nansum(out)
    return out


@njit
def nb_flatten1(nbhood, i, j, target):
    out = np.zeros_like(nbhood)
    if np.isnan(nbhood[i, j]):
        return out

    if nbhood[i, j] <= target - (out.size - 1):
        out[:, :] = -1
    elif nbhood[i, j] >= target + (out.size - 1):
        out[:, :] = 1

    out[:, :] *= (~np.isnan(nbhood)).astype(np.float32)
    out[i, j] = 0
    out[i, j] = -np.nansum(out)
    return out


@njit
def nb_flatten2(nbhood, i, j, target):
    non_nan = (~np.isnan(nbhood)).astype(np.float32)
    nnsum = np.nansum(non_nan)
    diff_v = nbhood[i, j] - target
    split_v = (diff_v / (nnsum - 1))
    out = split_v * non_nan
    out[i, j] = -diff_v
    return out


@njit
def nb_wrapper(nb_func, arr, i, j, target, size, out):
    if np.isnan(arr[i, j]):
        return False
    i0 = i - size
    i1 = i + size + 1
    j0 = j - size
    j1 = j + size + 1
    if i0 < 0: i0 = 0
    if j0 < 0: j0 = 0
    if i1 > arr.shape[0]: i1 = arr.shape[0]
    if j1 > arr.shape[1]: j1 = arr.shape[1]

    nbhood = arr[i0:i1, j0:j1]
    if np.isnan(nbhood).sum() == nbhood.size - 1:
        return False
    out[i0:i1, j0:j1] = out[i0:i1, j0:j1] + nb_func(nbhood, i-i0, j-j0, target)
    return True


@njit
def growing_nb_wrapper(nb_func, arr, i, j, target, max_size, out):
    if np.isnan(arr[i, j]):
        return False
    for size in range(1, max_size + 1):
        ret = nb_wrapper(nb_func, arr, i, j, target, size, out)
        if ret:
            return True
    return False


@njit
def apply_nbhood_gens(nb_func, arr, target, nb_max_size, num_gens):
    N = arr.shape[0]
    M = arr.shape[1]

    for g in range(num_gens):
        gen_diff = np.zeros((N, M), dtype=np.float32)

        for i in range(N):
            for j in range(M):
                growing_nb_wrapper(nb_func, arr, i, j, target, nb_max_size, gen_diff)

        arr[:, :] += gen_diff
    return arr

