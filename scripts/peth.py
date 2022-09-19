"""Plot peri-event time histograms (PETHs)."""

import logging

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import tqdm.contrib.logging
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="peth", version_base=None)
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}".format(OmegaConf.to_yaml(cfg)))

    # Update display parameters
    sns.set_theme(**cfg.seaborn)
    plt.rcParams.update(cfg.matplotlib)

    # Convert config parameters

    for neuron in tqdm.tqdm(cfg.neurons):
        # Re-direct logs within TQDM to not interfere with progress-bar
        with tqdm.contrib.logging.logging_redirect_tqdm():
            log.debug("Processing neuron: {}".format(neuron))
            plot_peth(dict(neuron), cfg)


def plot_peth(neuron: dict, cfg: DictConfig) -> None:
    """Plot peri-event time histogram (PETH) for a single neuron.

    :param neuron: dict with info to identify a neuron
    """
    assert isinstance(neuron, dict)
    assert "electrode" in neuron
    assert "pedestal" in neuron
    assert "unit" in neuron
    assert "session" in neuron


if __name__ == "__main__":
    main()
