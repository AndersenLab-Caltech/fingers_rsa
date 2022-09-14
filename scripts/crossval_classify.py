"""Classify task variable (such as movement)."""

import pynwb
import pandas as pd
import numpy as np
import xarray as xr

import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import cross_val_predict
from statsmodels.stats.weightstats import DescrStatsW

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import datetime
import logging
import os
import pathlib
from typing import List


log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="crossval_classify")
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}", OmegaConf.to_yaml(cfg))
    log.debug("Working directory: {}", os.getcwd())
    plt.rcParams.update(cfg.matplotlib)

    # Convert other config parameters
    time_bin = np.array([cfg.window.start, cfg.window.start + cfg.window.length])
    data_folder = pathlib.Path(get_original_cwd()).joinpath('data', cfg.task.dandiset)

    # Classifier to use for upcoming cross-validation
    clf = hydra.utils.instantiate(cfg.classifier)
    # Build of list of decoding accuracy
    results_df_list = []
    ds_list = []
    lc_list = []
    for session in cfg.task.sessions:
        # File-name format: https://dandi.readthedocs.io/en/latest/cmdline/organize.html
        nwb_path = data_folder.joinpath(
            f'sub-{cfg.array.subject}',
            f'sub-{cfg.array.subject}_ses-{session}_ecephys.nwb',
        )
        with pynwb.NWBHDF5IO(nwb_path, mode='r') as nwb_file:
            nwb = nwb_file.read()
        log.debug('Loaded NWB file:\n{}'.format(nwb))


if __name__ == "__main__":
    main()
