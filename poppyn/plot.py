import numpy as np
from matplotlib import pyplot as plt


def round_sig_fig(x, n):
    return np.round(x, -int(np.floor(np.log10(x))) + (n - 1))


def plot_data(arr, slice_name=None, pop_target=None, scale=None, max_in=None, hide_text=False):
    print("Generating plot")

    # Create the figure and axis
    height, width = arr.shape
    dpi = 100  # Adjust the DPI value as needed
    hin, win = height/dpi, width/dpi
    if max_in is not None and max_in < max(hin, win):
        hin, win = hin * max_in / max(hin, win), win * max_in / max(hin, win)
    fig, ax = plt.subplots(figsize=(win, hin), dpi=dpi)

    # Plot the image without borders
    ax.imshow(
        1.0 - arr,
        extent=[0, width, 0, height],
        origin='upper',
        cmap="gray_r",
        interpolation='nearest',
        aspect='equal',
    )

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove all padding and margins
    ax.margins(0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # Display the image with no borders or padding
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    for spine in ax.spines.values():
        spine.set_visible(False)

    if not hide_text:
        slice_name = "the World" if not slice_name else slice_name
        # Add text to the image
        font_properties = {
            'family': 'sans-serif',  # Font family (e.g., 'serif', 'sans-serif', 'monospace')
            'style': 'normal',  # Font style ('normal', 'italic', 'oblique')
            'weight': 'bold',  # Font weight ('normal', 'bold', 'light', 'heavy', 'ultrabold')
            'size': 16,  # Font size
            'color': 'white',  # Text color,
            'alpha': 0.4
        }
        text = f"A Pixel For Every {pop_target:,} People: {slice_name}"  # Text to be added
        ax.text(16, height - 16, text, fontdict=font_properties, ha='left', va='top')

        font_properties = {**font_properties, 'weight': 'normal', 'size': 12}
        text = (f"A total of {np.nansum(arr == 1.0).astype(int):,} white pixels have been placed, each representing {pop_target:,} people.\n"
                f"This is a geographic $approximation$ and should not be considered totally accurate; \n"
                f"  - In more dense areas, the population has been spread out;\n"
                f"  - In less dense areas, the population has been grouped together. See code for details.\n"
                f"On this {width}x{height} image, a pixel is around {int(round_sig_fig(scale, 2))}km$^2$.")
        ax.text(16, height - 40, text, fontdict=font_properties, ha='left', va='top')

        font_properties = {**font_properties, 'size': 10}
        text = (f"Dataset: WorldPop - DOI 10.5258/SOTON/WP00647\n"
                f"Code: Poppyn - github.com/naspli/poppyn\n"
                f"Creator: @naspli || u/_naspli")
        ax.text(16, 8, text, fontdict=font_properties, ha='left', va='bottom')
    return fig
