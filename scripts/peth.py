"""Plot peri-event time histograms (PETHs)."""

import functools
import logging
import os.path
import pathlib
import typing

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
import seaborn as sns
import tqdm
import tqdm.contrib.logging
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from scipy import stats

from fingers_rsa import nwb_utils

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="peth", version_base=None)
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}".format(OmegaConf.to_yaml(cfg)))

    # Update display parameters
    sns.set_theme(**cfg.seaborn)
    plt.rcParams.update(cfg.matplotlib)

    for neuron in tqdm.tqdm(cfg.neurons):
        # Re-direct logs within TQDM to not interfere with progress-bar
        with tqdm.contrib.logging.logging_redirect_tqdm():
            log.debug("Processing neuron: {}".format(neuron))
            plot_peth(**neuron, cfg=cfg)


def plot_peth(session: str, unit: typing.Any, cfg: DictConfig) -> None:
    """Plot peri-event time histogram (PETH) for a single neuron.

    :param session: which NWBFile session to load.
    :param unit: which unit to plot.
    :param cfg: Hydra config object
    """
    # Convert config parameters as needed
    data_folder = pathlib.Path(get_original_cwd()).joinpath("data", cfg.task.dandiset)
    nwb_path = data_folder.joinpath(
        f"sub-{cfg.array.subject}",
        f"sub-{cfg.array.subject}_ses-{session}_ecephys.nwb",
    )

    # Load NWB spike times, aligned to trial.
    border = 3  # TODO: Make this a config parameter
    with pynwb.NWBHDF5IO(nwb_path, mode="r") as nwb_file:
        nwb = nwb_file.read()
        # nwb_file must remain open while `nwb` object is in use.
        # Sub-select units to plot
        trial_spike_times_df = nwb_utils.align_spike_times_to_trials(
            nwb,
            start=cfg.plot.window.start - border,
            end=cfg.plot.window.end + border,
            include_units=[unit],
        )
        trial_labels: pd.Series = nwb.trials.to_dataframe()[cfg.task.condition_column]

    # Convert spike times to firing rates
    sample_times = np.arange(
        cfg.plot.window.start,
        cfg.plot.window.end + cfg.plot.window.step,
        cfg.plot.window.step,
    )
    firing_rates = [
        convolve_func(
            sample_time, trial_spike_times_df.loc[:, unit], cfg.plot.smooth_width
        )
        for sample_time in sample_times
    ]
    firing_rates_df = pd.concat(firing_rates, axis=1, keys=sample_times, names=["time"])

    # Reshape wide dataframe into long format for plotting
    firing_rates_melt_df = firing_rates_df.melt(
        value_name="firing rate", ignore_index=False
    )
    # Merge on index to add trial labels
    firing_rates_melt_df["finger"] = trial_labels

    ax = sns.lineplot(
        data=firing_rates_melt_df,
        x="time",
        y="firing rate",
        hue=cfg.task.condition_column,
        hue_order=cfg.task.condition_order,
    )
    title = f"{session}_unit-{unit}"
    ax.set_title(title)

    # Save figure
    ax.get_figure().savefig(title)
    log.info("Saved plot: {}".format(os.path.abspath(title)))
    # Clear figure for next plot
    ax.clear()


def convolve_func(
    sample_time: float, spike_times: pd.Series, kernel_std: float
) -> pd.Series:
    """Convolve Gaussian smoothing kernel with spike times.

    :param sample_time: time to center the kernel on.
    :param spike_times: spike times to convolve with kernel.
    :param kernel_std: standard deviation of Gaussian kernel.
    :returns: smoothed firing rates at the sample times.
    """
    kernel = functools.partial(stats.norm.pdf, loc=sample_time, scale=kernel_std)
    firing_rates = spike_times.apply(kernel).apply(sum)
    return firing_rates


if __name__ == "__main__":
    main()
