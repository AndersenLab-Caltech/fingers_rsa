"""Classify task variable using cross-validation and plot the confusion matrix.
"""
import logging
import os
import pathlib
from typing import Tuple

import hydra
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from statsmodels.stats.weightstats import DescrStatsW

from fingers_rsa import nwb_utils, plot_utils

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="crossval_classify", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}".format(OmegaConf.to_yaml(cfg)))
    log.debug("Working directory: {}".format(os.getcwd()))
    plt.rcParams.update(cfg.matplotlib)

    # Convert config parameters as needed
    data_folder = pathlib.Path(get_original_cwd()).joinpath("data", cfg.task.dandiset)

    # Build up list of cross-validated predictions
    results_df_list = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(cv_results)(session, data_folder, cfg)
        for session in cfg.task.sessions
    )
    all_results_df: pd.DataFrame = pd.concat(
        results_df_list, keys=cfg.task.sessions, names=["session"]
    )

    accuracy, std = log_summary_metrics(
        all_results_df[cfg.task.condition_column], all_results_df.predicted
    )

    # Show confusion matrix
    title = (
        "Aggregate confusion matrix for {subject}, {var_name}\n"
        + "{trials} trials over {sessions} sessions, "
        + "{phase}: {time_bin}\n"
        + "Cross-validated accuracy: {accuracy:.0%} +/- {std:.0%}"
    ).format(
        subject=cfg.array.subject_display_id,
        var_name=cfg.task.condition_column,
        trials=len(all_results_df),
        sessions=len(cfg.task.sessions),
        phase=cfg.task.phase,
        time_bin=[cfg.window.start, cfg.window.start + cfg.window.length],
        accuracy=accuracy,
        std=std,
    )
    fig, ax = plt.subplots()
    plot_utils.plot_confusion_matrix(
        all_results_df[cfg.task.condition_column],
        all_results_df.predicted,
        cfg.task.condition_order,
        title=title,
        cmap=mpl.rcParams["image.cmap"],
        ax=ax,
        include_values=cfg.confusion_metrics.include_values,
        values_format=cfg.confusion_metrics.values_format,
    )
    fig.savefig(f"crossval_confusion_matrix_{cfg.task.condition_column}")

    plt.show()


def cv_results(
    session: str, data_folder: pathlib.Path, cfg: DictConfig
) -> pd.DataFrame:
    """Calculates cross-validation results for a single session.

    Wraps logic so this can easily be run within a joblib.Parallel session.

    :param session: identifier for session to analyze
    :param data_folder: folder with NWB files
    :param cfg: Hydra config object

    :returns: DataFrame with true and crossval-predicted labels for each trial
    """
    nwb_path = data_folder.joinpath(
        f"sub-{cfg.array.subject}",
        f"sub-{cfg.array.subject}_ses-{session}_ecephys.nwb",
    )
    log.debug("Loading NWB file: {}".format(nwb_path))
    trial_spike_counts, trial_labels = read_trial_features(nwb_path, cfg)

    # Classifier and cross-validation methods
    clf = hydra.utils.instantiate(cfg.classifier)
    cv = hydra.utils.instantiate(cfg.crossvalidation)
    trial_pred = cross_val_predict(
        clf,
        trial_spike_counts.values,
        trial_labels.values,
        cv=cv,
        n_jobs=-1,
        method="predict",
    )
    results_df = trial_labels.to_frame()
    results_df["predicted"] = trial_pred

    log.debug("Finished processing NWB file: {}".format(nwb_path))
    return results_df


def read_trial_features(
    nwb_path: os.PathLike, cfg: DictConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    """Reads trial spike counts and labels from NWB file.

    :param nwb_path: path to NWB file
    :param cfg: Hydra config object
    :returns: tuple of (spike counts, labels)
    """
    with pynwb.NWBHDF5IO(nwb_path, mode="r") as nwb_file:
        nwb = nwb_file.read()
        # nwb_file must remain open while `nwb` object is in use.

        trial_spike_counts = nwb_utils.count_trial_spikes(
            nwb,
            start=cfg.window.start,
            end=cfg.window.start + cfg.window.length,
        )
        trial_labels: pd.Series = nwb.trials.to_dataframe()[cfg.task.condition_column]

    return trial_spike_counts, trial_labels


def log_summary_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    """Logs metrics for classification predictions across sessions.

    :param y_true: true labels
    :param y_pred: predicted labels
    :returns: tuple of (accuracy, weighted standard deviation)
    """
    # Get average accuracy and standard deviation (weighted by trial counts)
    # across sessions
    is_predict_correct = y_true == y_pred
    summary = is_predict_correct.groupby(level="session").agg(["mean", "count"])
    wdf = DescrStatsW(summary["mean"], weights=summary["count"], ddof=1)
    accuracy = accuracy_score(y_true, y_pred)
    log.info(
        "Accuracy: {:.0%} +/- {:.0%} over {:d} sessions.".format(
            accuracy,
            wdf.std,
            len(summary),
        )
    )
    np.testing.assert_almost_equal(
        accuracy, wdf.mean, err_msg="accuracy calculations should match"
    )

    return accuracy, wdf.std


if __name__ == "__main__":
    main()
