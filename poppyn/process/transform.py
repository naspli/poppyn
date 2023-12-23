import numpy as np


def logscale(arr):
    return np.log10(arr + 1)


def logsteps(arr):
    return np.floor(2 * np.log10(arr + 1)) * 0.5


def steps(arr, step_size=1000):
    return np.floor(arr / step_size) * step_size


def binary(arr, step_size=1000):
    return np.minimum(np.floor(arr / step_size) * step_size, step_size)
