import matplotlib.pyplot as plt

from poppyn.inout import *
from poppyn.plot import *
from poppyn.process.combine import *
from poppyn.process.ordered import *

slice_name = "GB"
max_resolution = 1000
pop_target = 10_000

cache_key = f"{slice_name}_{max_resolution}_{pop_target}"

if get_data_path(get_slice_filename(cache_key)).exists():
    bin_arr = load_slice(cache_key)
else:
    og_arr = load_slice(slice_name)
    red_arr = reduce_resolution(og_arr, max_size=max_resolution, min_pop=0.01 * pop_target)
    bin_arr = select_and_flatten_largest_points(red_arr, pop_target)
    save_slice(cache_key, bin_arr)

arr = np.nan_to_num(bin_arr, nan=-0.1)
show_data(arr)

plt.show(block=True)