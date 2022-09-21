"""Decompose representational structure into component models over time."""

import glob
import logging
import os
import pathlib
import typing

import hydra
import numpy as np
import pandas as pd
import rsatoolbox
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="rda", version_base=None)
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}".format(OmegaConf.to_yaml(cfg)))

    # Convert config parameters as needed
    data_rdm_files = glob.glob(cfg.rdm_files)
    filename_prefix = filename(cfg)

    # Load in models. These should be saved as RDM files, too.
    model = load_models(cfg.rsa.models)

    # Read in the RDM files
    data_rdm_list = [rsatoolbox.rdm.load_rdm(rdm_file) for rdm_file in data_rdm_files]
    window_length_list = [
        rdm.rdm_descriptors["window_length"][0] for rdm in data_rdm_list
    ]
    window_center_list = [
        rdm.rdm_descriptors["window_center"][0] for rdm in data_rdm_list
    ]

    # Validate data
    np.testing.assert_array_equal(
        window_length_list,
        window_length_list[0],
        err_msg="Window lengths should be the same for all RDMs",
    )

    # Group RDMs by window center to concatenate
    rdm_df = pd.DataFrame({"window_center": window_center_list, "rdm": data_rdm_list})
    data_rdms_ser = rdm_df.groupby("window_center")["rdm"].apply(rsatoolbox.rdm.concat)

    # Validate data
    rdm_counts = data_rdms_ser.apply(lambda rdm: rdm.n_rdm)
    np.testing.assert_array_equal(
        rdm_counts,
        rdm_counts.iloc[0],
        err_msg="Expected same number of RDMs for each window center",
    )
    assert (
        rdm_counts > 1
    ).all(), "Need more than one RDM to calculate RSA confidence intervals"

    # RDA: run representational similarity analysis for each window center
    method = "cosine_cov"  # Method for normalization
    results_ser = data_rdms_ser.apply(
        lambda data_rdms: rsatoolbox.inference.eval_bootstrap_rdm(
            models=model,
            data=data_rdms,
        )
    )

    # Convert RSA Result objects to evaluations
    evaluations_ser = results_ser.apply(result_to_dataframe)
    evaluations_wide_df = pd.concat(
        evaluations_ser.tolist(), keys=evaluations_ser.index
    )
    evaluations_df = evaluations_wide_df.melt(value_name=method, ignore_index=False)

    # Plot results
    ax = sns.lineplot(
        data=evaluations_df,
        x="window_center",
        y=method,
        hue="model",
        # TODO: double-check error-bar matches previous version
        errorbar="sd",  # standard deviation of bootstrap -> confidence interval
    )

    # Save plot
    plot_filename = filename_prefix
    ax.figure.savefig(plot_filename)
    log.info("Saved plot to file: {}".format(os.path.abspath(plot_filename)))


def result_to_dataframe(result: rsatoolbox.inference.Result) -> pd.DataFrame:
    """Convert rsatoolbox Result to plottable DataFrame.

    :param result: RSA result object to convert
    :returns: [sample, model]-shaped DataFrame with evaluation values
    """
    df = pd.DataFrame(
        result.evaluations,
        columns=[model.name for model in result.models],
    )
    df.columns.name = "model"
    df.index.name = "bootstrap sample"

    return df


def load_models(
    include_models: typing.List[str],
    model_dir: os.PathLike = "models",
) -> rsatoolbox.model.Model:
    """Load in model RDMs from HDF5 (.h5 / .hdf5) files.

    :param include_models: list of model names (filename prefixes) to include
    :param model_dir: directory containing model RDMs
    :returns: list of model RDMs
    """
    model_files = pathlib.Path(model_dir).glob("*.h*5")
    rdms = [
        rsatoolbox.rdm.load_rdm(str(model_file))
        for model_file in model_files
        if model_file.stem in include_models
    ]
    rdms = rsatoolbox.rdm.concat(rdms)
    model = rsatoolbox.model.ModelWeighted("nnls", rdms)

    return model


def filename(cfg: DictConfig) -> str:
    """Generate filename for representational dynamics analysis.

    :param cfg: Hydra config object, including task info
    :return: filename for RDA
    """
    return ("sub-{subject}_rda").format(
        subject=cfg.array.subject,
    )


if __name__ == "__main__":
    main()
