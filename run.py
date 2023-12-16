import matplotlib.pyplot as plt

from poppyn.inout import *
from poppyn.plot import *
from poppyn.process.combine import *
from poppyn.process.ordered import *

# try ordering everything min->max
# then "select" the points 1-by-1 instead
# once selected, dump extra points to neighbours (discourage ocean/zeros but make impossible to already selected points)
# once drop below threshold, do a neighbour blur and reorder again
# repeat until you have selected total_pop / target points


arr = load_slice()
show_data_logscale(arr)

arr = reduce_resolution(arr, 100)
show_data_logscale(arr)

arr = select_and_flatten_largest_points(arr, 1000000)
show_data(arr)

# show_data_logscale(arrx)
#
# arr2 = apply_nbhd_gens(arrx, nb_flatten, nb_size=3, target=1000000, num_gens=100)
# show_data(arr2)
#
# arr3 = apply_nbhd_gens(arr2, nb_flatten, nb_size=10, target=1000000, num_gens=100)
# show_data(arr3)
plt.show(block=True)
