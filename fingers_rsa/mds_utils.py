"""Visualization helper functions, especially for multi-dimensional scaling."""

from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import prince
from matplotlib.patches import Ellipse


def make_mds_plot(
    mds: np.array,
    ax: mpl.axes.Axes,
    condition_order: List[str],
    colors: Optional[List[str]] = None,
) -> None:
    # mds has shape (n_sessions, n_conditions, n_mds_dims)
    # condition_order: order corresponding to mds n_conditions
    mds_aligned = prince.GPA().fit_transform(mds)
    make_mds_plot_prealigned(mds_aligned, ax, condition_order, colors)


def make_mds_plot_prealigned(
    mds_aligned: np.array,
    ax: mpl.axes.Axes,
    condition_order: List[str],
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    raw: bool = False,
    **plot_kwargs,
) -> None:
    """Helper function, assuming that MDS is already GPA-aligned.

    Useful when you want to align a full dataset, then plot subsets separately
    (e.g., a pre/post-effect like Kieliba/Makin 2021 Fig 3D)
    """
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    for cond_idx, cond in enumerate(condition_order):
        x = mds_aligned[:, cond_idx, 0]
        y = mds_aligned[:, cond_idx, 1]
        color = colors[cond_idx]
        marker = markers[cond_idx] if (markers is not None) else None
        if raw:
            ax.scatter(x, y, c=color, label=cond, marker=marker, **plot_kwargs)
        else:
            # Plot ellipse boundary and center
            confidence_ellipse(x, y, ax, n_std=1, edgecolor=color, **plot_kwargs)
            ax.scatter(
                x.mean(),
                y.mean(),
                color=color,
                label=cond,
                marker=marker,
                **plot_kwargs,
            )
        ax.annotate(
            cond, (x.mean(), y.mean()), xytext=(15, 15), textcoords="offset points"
        )

    # Format plot
    ax.margins(0.15)
    ax.axis("scaled")
    # Change spines/ticks to size-bar
    draw_sizebar(ax)


def draw_sizebar(ax: mpl.axes.Axes, size=1, units=None) -> None:
    # Draw sizebars in the style of https://www.nature.com/articles/nn.4038/figures/3
    # Remove un-used spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set parameters for both axes
    ax.tick_params(direction="in")
    # Make minor ticks invisible
    ax.tick_params(which="minor", length=0, labelsize="medium")

    # Draw x-scale bar
    xlim = ax.get_xlim()
    x_bar_start = np.percentile(xlim, 10)
    x_bounds = (x_bar_start, x_bar_start + size)
    ax.spines["bottom"].set_bounds(x_bounds)
    # Create only 2 ticks at the bounds, without labels
    ax.set_xticks(x_bounds)
    ax.set_xticklabels([])
    # Set an invisible minor tick in between the bounds
    ax.set_xticks(np.mean(x_bounds, keepdims=True), minor=True)
    # Give the invisible minor tick a label with the length
    bar_size = np.diff(x_bounds)[0]
    bar_label = str(bar_size)
    if units is not None:
        bar_label += f" [{units}]"
    ax.set_xticklabels([bar_label], minor=True)

    # Repeat for y-axis
    ylim = ax.get_ylim()
    x_bar_offset = x_bar_start - xlim[0]  # Use same offset, assuming scaled axes
    y_bar_start = ylim[0] + x_bar_offset
    y_bounds = (y_bar_start, y_bar_start + size)
    ax.spines["left"].set_bounds(y_bounds)
    ax.set_yticks(y_bounds)
    ax.set_yticklabels([])
    ax.set_yticks(np.mean(y_bounds, keepdims=True), minor=True)
    ax.set_yticklabels([bar_label], minor=True)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Taken from:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
