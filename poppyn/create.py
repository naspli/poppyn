from poppyn.inout import get_data_path, get_slice_filename, load_slice, save_data
from poppyn.process.combine import reduce_resolution
from poppyn.process.ordered import select_and_flatten_largest_points


def create_binary_population_map(slice_name, max_resolution, pop_target, slice_idx=None, force_rerun=False):
    cache_key = f"{slice_name.replace(' ', '')}_{max_resolution}_{pop_target}"
    cache_path = get_data_path(get_slice_filename(cache_key))

    if not force_rerun and cache_path.exists():
        return load_slice(cache_key)

    og_arr = load_slice(slice_name, idx=slice_idx)
    red_arr = reduce_resolution(og_arr, max_size=max_resolution, min_pop=0.01 * pop_target)
    bin_arr = select_and_flatten_largest_points(red_arr, pop_target)
    save_data(cache_path, bin_arr)
    return bin_arr
