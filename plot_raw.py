import argparse

import matplotlib.pyplot as plt

from poppyn.inout import load_or_generate_slice
from poppyn.plot import plot_data
from poppyn.process.transform import logscale


def main():
    parser = argparse.ArgumentParser("Show raw WorldPop Data.")
    parser.add_argument("--slice_name", default="",
                        help="Use a pre-defined slice of the World map (see statics.py)")
    args = parser.parse_args()

    arr = load_or_generate_slice(args.slice_name)
    log_arr = logscale(arr)
    plot_data(
        log_arr,
        max_in=10,
        hide_text=True
    )
    plt.show(block=True)


if __name__ == "__main__":
    main()
