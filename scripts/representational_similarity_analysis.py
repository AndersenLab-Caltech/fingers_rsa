"""Calculate similarity between representational dissimilarity matrices."""

import glob
import logging
import os
import pathlib
import typing

import hydra
import matplotlib.pyplot as plt
import rsatoolbox
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="rsa", version_base=None)
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}".format(OmegaConf.to_yaml(cfg)))
    # Update plotting config parameters
    plt.rcParams.update(cfg.matplotlib)

    # Convert config parameters as needed
    data_rdm_files = glob.glob(cfg.rdm_files)
    filename_prefix = filename(cfg)

    # Read in the RDM files
    data_rdm_list = [rsatoolbox.rdm.load_rdm(rdm_file) for rdm_file in data_rdm_files]
    data_rdms = rsatoolbox.rdm.concat(data_rdm_list)

    # Load in models. These should be saved as RDM files, too.
    model_rdms = load_models()

    assert (
        data_rdms.n_rdm > 1
    ), "Need more than one RDM to calculate RSA confidence intervals"
    results: rsatoolbox.inference.Result = rsatoolbox.inference.eval_bootstrap_rdm(
        model_rdms,
        data_rdms,
        method=cfg.metrics.similarity,
    )
    # Save results to hdf5, including metadata
    results_filename = filename_prefix + ".hdf5"
    results.save(results_filename, file_type="hdf5", overwrite=True)
    log.info("Saved results to file: {}".format(os.path.abspath(results_filename)))

    rsatoolbox.vis.plot_model_comparison(
        results,
        test_pair_comparisons="golan",
    )
    # Save plot
    # matplotlib will automatically add the file extension, assuming
    # `plot_filename` doesn't have a period
    plot_filename = filename_prefix
    plt.savefig(plot_filename, bbox_inches="tight")
    log.info("Saved plot to file: {}".format(os.path.abspath(plot_filename)))

    plt.show()


def load_models() -> typing.List[rsatoolbox.model.Model]:
    model_rdm_files = glob.glob("models/*.h*5")

    # Load model rdms
    model_list = []
    for model_rdm_file in model_rdm_files:
        rdm = rsatoolbox.rdm.load_rdm(model_rdm_file)
        # Assume the model RDMs' conditions are already sorted in the same
        # order as the data RDMs' conditions

        model_name = pathlib.Path(model_rdm_file).stem
        model_list.append(rsatoolbox.model.ModelFixed(model_name, rdm))

    return model_list


def filename(cfg: DictConfig) -> str:
    """Generate filename for representational similarity analysis.

    :param cfg: Hydra config object, including task info
    :return: filename for RSA
    """
    return ("sub-{subject}_rsa").format(
        subject=cfg.array.subject,
    )


if __name__ == "__main__":
    main()
