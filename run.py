import matplotlib.pyplot as plt

from inout import load_slice
from plot import show_data
from process import apply_nbhd_gens, nb_flatten, nb_trickle

arr = load_slice("London")
show_data(arr)

arr2 = apply_nbhd_gens(arr, nb_flatten, target=10000, num_gens=1000)
show_data(arr2)

plt.show(block=True)