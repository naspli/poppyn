import matplotlib.pyplot as plt

from inout import load_slice
from plot import show_data, show_data_logscale
from process import apply_nbhd_gens, nb_flatten, nb_trickle
from tools import reduce_resolution

arr = load_slice("GB")
show_data_logscale(arr)

arrx = reduce_resolution(arr, 3)
show_data_logscale(arrx)

# arr2 = apply_nbhd_gens(arrx, nb_flatten, nb_size=3, target=1000000, num_gens=100)
# show_data(arr2)
#
# arr3 = apply_nbhd_gens(arr2, nb_flatten, nb_size=10, target=1000000, num_gens=100)
# show_data(arr3)


plt.show(block=True)