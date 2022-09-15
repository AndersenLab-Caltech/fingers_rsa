"""Plotting utilities"""

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from typing import List


def plot_confusion_matrix(
        y_true: pd.Series,
        y_pred: pd.Series,
        label_order: List,
        title: str = '',
        **plot_kwargs,
):
    """Helper function to plot confusion matrix."""
    conf_mat = confusion_matrix(y_true, y_pred, labels=label_order, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=label_order)
    disp.plot(**plot_kwargs)
    disp.im_.set_clim(0, 1)
    # Limit number of ticks, for less clutter
    disp.im_.colorbar.ax.locator_params(nbins=3)
    disp.im_.colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    disp.figure_.suptitle(title)
