import argparse

import matplotlib.pyplot as plt
import numpy as np

from poppyn.create import create_binary_population_map
from poppyn.inout import save_image, get_scale
from poppyn.plot import plot_data


def main():
    parser = argparse.ArgumentParser("Generate Binary Population Map and Save Image. Depends on WorldPop Data.")
    parser.add_argument("--slice_name", default="",
                        help="Use a pre-defined slice of the World map (see statics.py) OR name your own slice_idx")
    parser.add_argument("--slice_idx", default=None, type=int, nargs=4,
                        help="Y0 Y-1 X0 X-1 indexes of your custom slice of WorldPop 1KM dataset")
    parser.add_argument("--max_resolution", type=int, default=1920,
                        help="Downsample so longest axis is this long")
    parser.add_argument("--pop_target", type=int, default=1_000_000,
                        help="What population should each pixel represent?")
    parser.add_argument("--parallel_chunks", type=int, default=None,
                        help="Split array into this many chunks before calculation?")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Ignore cached data and recreate map")
    parser.add_argument("--show", action="store_true",
                        help="Instead of saving to disk, show the image on screen.")
    parser.add_argument("--hide_text", action="store_true",
                        help="Raw map without text. Don't forget to credit...")
    args = parser.parse_args()

    bin_arr = create_binary_population_map(
        args.slice_name,
        args.max_resolution,
        args.pop_target,
        slice_idx=args.slice_idx,
        force_rerun=args.force_rerun,
        parallel_chunks=args.parallel_chunks
    )
    plot_arr = np.nan_to_num(bin_arr, nan=-0.1)
    fig = plot_data(
        plot_arr,
        slice_name=args.slice_name,
        pop_target=args.pop_target,
        scale=get_scale(plot_arr, args.slice_name),
        hide_text=args.hide_text
    )

    if args.show:
        plt.show(block=True)
    else:
        cache_key = f"{args.slice_name.replace(' ', '')}_{args.max_resolution}_{args.pop_target}"
        save_image(cache_key, fig)


if __name__ == "__main__":
    main()
