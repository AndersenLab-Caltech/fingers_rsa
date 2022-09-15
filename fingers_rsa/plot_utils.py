"""Plotting utilities"""

from typing import List

import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    label_order: List,
    title: str = "",
    **plot_kwargs,
):
    """Plot confusion matrix on a [0, 100%] scale.

    :param y_true: true labels
    :param y_pred: predicted labels
    :param label_order: order of labels on axes
    :param title: title for plot
    :param plot_kwargs: additional keyword arguments to pass to
        ConfusionMatrixDisplay
    """
    conf_mat = confusion_matrix(y_true, y_pred, labels=label_order, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=label_order)
    disp.plot(**plot_kwargs)
    disp.im_.set_clim(0, 1)
    # Limit number of ticks, for less clutter
    disp.im_.colorbar.ax.locator_params(nbins=3)
    disp.im_.colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    disp.figure_.suptitle(title)
