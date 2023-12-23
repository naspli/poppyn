import argparse

from poppyn.inout import *
from poppyn.plot import *
from poppyn.process.combine import *
from poppyn.process.ordered import *


def create_binary_population_map(slice_name, max_resolution, pop_target, force_rerun=False):
    cache_key = f"{slice_name.replace(' ', '')}_{max_resolution}_{pop_target}"
    cache_path = get_data_path(get_slice_filename(cache_key))

    if not force_rerun and cache_path.exists():
        return load_slice(cache_key)

    og_arr = load_slice(slice_name)
    red_arr = reduce_resolution(og_arr, max_size=max_resolution, min_pop=0.01 * pop_target)
    bin_arr = select_and_flatten_largest_points(red_arr, pop_target)
    save_data(cache_path, bin_arr)
    return bin_arr


def main():
    slice_name = "Britain"
    max_resolution = 1000
    pop_target = 10_000
    force_rerun = False
    show = False

    bin_arr = create_binary_population_map(slice_name, max_resolution, pop_target, force_rerun=force_rerun)
    fig = show_data(bin_arr, slice_name=slice_name, pop_target=pop_target, scale=get_scale(bin_arr, slice_name))

    if show:
        plt.show(block=True)
    else:
        cache_key = f"{slice_name.replace(' ', '')}_{max_resolution}_{pop_target}"
        save_image(cache_key, fig)


if __name__ == "__main__":
    main()
