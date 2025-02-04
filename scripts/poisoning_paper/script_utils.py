"""
Helper functions for the privacy paper scripts.
"""

import os

from matplotlib.image import imread
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

# set plotting context
tex_fonts = {
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": True,
}

plt.style.use(["science", "no-latex"])
sns.set_palette("Dark2")
plt.rcParams.update(tex_fonts)

palette = sns.color_palette("Dark2")
sequential_colours = sns.color_palette(palette="Greens", n_colors=12)
colours = {
    "green": "C0",
    "orange": "C1",
    "purple": "C2",
    "pink": "C3",
    "light_green": "C4",
    "yellow": "C5",
    "brown": "C6",
    "grey": "C7",
}
lb_color = list(iter(sns.color_palette(palette="deep")))[-1]
ub_color = list(iter(sns.color_palette(palette="deep")))[-2]


def set_size(fraction=1.0, subplots=(1, 1), portrait=False, shrink_height=1.0):
    """
    Useful function for setting figure dimensions to avoid scaling in LaTeX from
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/.

    Parameters
    ----------
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width_pt = 469.754  # document width in pt, as revealed by \showthe\textwidth in latex

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    if portrait is False:
        fig_height_in = shrink_height * fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = shrink_height * fig_width_in / golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def get_size(fig, dpi=300):
    """Code from https://kavigupta.org/2019/05/18/Setting-the-size-of-figures-in-matplotlib/
    which fixes the super annoying matplotlib figure size handling."""
    with NamedTemporaryFile(suffix=".png") as f:
        fig.savefig(f.name, bbox_inches="tight", dpi=dpi)
        height, width, _channels = imread(f.name).shape
        return width / dpi, height / dpi


def apply_figure_size(fig, size, dpi=300, eps=1e-2, give_up=2, min_size_px=10):
    """Code from https://kavigupta.org/2019/05/18/Setting-the-size-of-figures-in-matplotlib/
    which fixes the super annoying matplotlib figure size handling."""
    target_width, target_height = size
    set_width, set_height = target_width, target_height  # reasonable starting point
    deltas = []  # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False


def make_dirs():
    """Make the required directories"""
    dirname = os.path.dirname(__file__)
    dirs = [".results", ".models", ".data", ".figures"]
    dirs = [os.path.join(dirname, d) for d in dirs]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    return dirs
