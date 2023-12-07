import numpy as np
from matplotlib import pyplot as plt


def fit_size(arr, long=10):
    return tuple(long * x / max(arr.shape) for x in reversed(arr.shape))


def show_data(arr):
    plt.figure(figsize=fit_size(arr))
    plt.imshow(arr, cmap="gray_r", interpolation='nearest', aspect='equal')
    plt.colorbar()
    plt.show(block=False)


def show_data_logscale(arr):
    plt.figure(figsize=fit_size(arr))
    arr_log = np.log10(arr + 1)
    plt.imshow(arr_log, cmap="gray_r", vmin=-0.5, interpolation='nearest', aspect='equal')
    plt.colorbar(label='log10 scale')
    plt.show(block=False)


def show_data_logsteps(arr):
    plt.figure(figsize=fit_size(arr))
    arr_steps = np.floor(2 * np.log10(arr + 1)) * 0.5
    plt.imshow(arr_steps, cmap="gray_r", vmin=-0.5, interpolation='nearest', aspect='equal')
    plt.colorbar(label='log10 scale')
    plt.show(block=False)


def show_data_steps(arr):
    plt.figure(figsize=fit_size(arr))
    arr_bin = np.floor(arr / 1000) * 1000
    plt.imshow(arr_bin, cmap="gray_r", interpolation='nearest', aspect='equal')
    plt.colorbar()
    plt.show(block=False)


def show_data_bin(arr):
    plt.figure(figsize=fit_size(arr))
    arr_bin = np.minimum(np.floor(arr / 1000) * 1000, 1000)
    plt.imshow(arr_bin, cmap="gray_r", interpolation='nearest', aspect='equal')
    plt.colorbar()
    plt.show(block=False)
