"""Generate representational dissimilarity matrices."""

import logging
import os
import pathlib

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import rsatoolbox
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from fingers_rsa import nwb_utils, rdm_utils

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="rdm")
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}".format(OmegaConf.to_yaml(cfg)))
    log.debug("Working directory: {}".format(os.getcwd()))
    plt.rcParams.update(cfg.matplotlib)

    # Convert config parameters as needed
    data_folder = pathlib.Path(get_original_cwd()).joinpath("data", cfg.task.dandiset)
    rdm_path = pathlib.Path(rdm_utils.filename(cfg))

    for session in cfg.task.sessions:
        nwb_path = data_folder.joinpath(
            f"sub-{cfg.array.subject}",
            f"sub-{cfg.array.subject}_ses-{session}_ecephys.nwb",
        )
        trial_spike_counts, trial_labels = nwb_utils.read_trial_features(nwb_path, cfg.task.condition_column, start=cfg.window.start, end=cfg.window.start + cfg.window.length)

    log.debug(f'RDMs:\n{data_rdms}')

    # Save out RDMs for loading from future scripts.
    log.info('Saving RDMs to: {}'.format(rdm_path.resolve()))
    data_rdms.save(rdm_path, file_type=cfg.rdm.file_type, overwrite=True)

    # Save out RDM plots
    num_pattern_groups = len(cfg.task.condition_order) - int(cfg.task.null_condition in cfg.task.condition_order)
    fig = rsatoolbox.vis.show_rdm(
        data_rdms,
        pattern_descriptor=cfg.task.condition_column,
        cmap=mpl.rcParams['image.cmap'],
        rdm_descriptor='session',
        show_colorbar='figure',
        vmin=0,
        num_pattern_groups=num_pattern_groups,
    )
    log.info(f'Saving RDM plot to: {}'.format(rdm_path.resolve()))
    fig.savefig(rdm_path)

    plt.show()


if __name__ == "__main__":
    main()
