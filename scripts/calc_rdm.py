"""Generate representational dissimilarity matrices."""

import logging
import os
import pathlib

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rsatoolbox
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from fingers_rsa import nwb_utils, rdm_utils

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="rdm", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}".format(OmegaConf.to_yaml(cfg)))
    log.debug("Working directory: {}".format(os.getcwd()))
    plt.rcParams.update(cfg.matplotlib)

    # Convert config parameters as needed
    data_folder = pathlib.Path(get_original_cwd()).joinpath("data", cfg.task.dandiset)
    rdm_name = rdm_utils.filename(cfg)

    nwb_path = data_folder.joinpath(
        f"sub-{cfg.array.subject}",
        f"sub-{cfg.array.subject}_ses-{cfg.session}_ecephys.nwb",
    )
    trial_spike_counts, trial_labels = nwb_utils.read_trial_features(
        nwb_path,
        cfg.task.condition_column,
        start=cfg.window.start,
        end=cfg.window.start + cfg.window.length,
    )
    rdm = generate_rdm(trial_spike_counts.values, trial_labels.values, cfg)

    log.debug(f"RDMs:\n{rdm}")

    # Save out RDMs for loading from future scripts.
    rdm_file_name = rdm_name + "." + cfg.rdm_file_type
    log.info("Saving RDMs to: {}".format(rdm_file_name))
    rdm.save(rdm_file_name, file_type=cfg.rdm_file_type, overwrite=True)

    # Save out RDM plots
    num_pattern_groups = len(cfg.task.condition_order) - int(
        cfg.task.null_condition in cfg.task.condition_order
    )
    fig, _, _ = rsatoolbox.vis.show_rdm(
        rdm,
        pattern_descriptor=cfg.task.condition_column,
        cmap=mpl.rcParams["image.cmap"],
        rdm_descriptor="session",
        show_colorbar="figure",
        vmin=0,
        num_pattern_groups=num_pattern_groups,
    )
    rdm_plot_name = rdm_name + "." + plt.rcParams["savefig.format"]
    log.info("Saving RDM plot to: {}".format(rdm_plot_name))
    fig.savefig(rdm_plot_name)

    plt.show()


def generate_rdm(
    measurements: np.ndarray,
    trial_labels: np.ndarray,
    cfg: DictConfig,
    # descriptors: typing.Dict[str, str],
    # observation_descriptors: typing.Dict[str, np.ndarray],
    # TODO, refactor out the config arguments.
    #  E.g., non-null could just come explicit pass-in list of parameters
    ignore_null_condition: bool = True,
) -> rsatoolbox.rdm.RDMs:
    """Generate RDM from  xarray Dataset into rsatoolbox Dataset."""

    # TODO: could also do this filtering at the NWBFile level
    if ignore_null_condition:
        null_trial_mask = trial_labels == cfg.task.null_condition
        measurements = measurements[~null_trial_mask]
        trial_labels = trial_labels[~null_trial_mask]

    observation_descriptors = {cfg.task.condition_column: trial_labels}
    descriptors = {"task_name": cfg.task.name, "subject": cfg.array.subject}

    # Create rsatoolbox dataset
    dataset = rsatoolbox.data.Dataset(
        measurements=measurements,
        descriptors=descriptors,
        obs_descriptors=observation_descriptors,
    )

    rdm = rsatoolbox.rdm.calc_rdm(
        dataset, method=cfg.metrics.distance, descriptor=cfg.task.condition_column
    )
    rdm.rdm_descriptors["session"] = [cfg.session]
    condition_order = list(cfg.task.condition_order)
    if ignore_null_condition:
        condition_order.remove(cfg.task.null_condition)
    rdm.sort_by(**{cfg.task.condition_column: condition_order})

    return rdm


if __name__ == "__main__":
    main()
