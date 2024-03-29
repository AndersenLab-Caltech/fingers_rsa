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
import sklearn.linear_model
import sklearn.preprocessing
import tqdm
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="rda", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}".format(OmegaConf.to_yaml(cfg)))

    # Convert config parameters as needed
    data_rdm_files = glob.glob(cfg.rdm_files)
    assert len(data_rdm_files) > 0, "No RDM files found at: {}.".format(cfg.rdm_files)
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

    # RDA: fit linear regression for RDM calculated at each window_center
    results_ser = data_rdms_ser.apply(
        lambda data_rdms: fit_bootstrap(model=model, data_rdms=data_rdms)
    )
    # Convert to long-format for plotting
    results_wide_df = pd.concat(results_ser.tolist(), keys=results_ser.index)
    value_name = "model_weight"
    results_df = results_wide_df.melt(value_name=value_name, ignore_index=False)

    # Plot results
    ax = sns.lineplot(
        data=results_df,
        x="window_center",
        y=value_name,
        hue="model",
        hue_order=cfg.rsa.models,
        errorbar="sd",  # standard deviation of bootstrap -> confidence interval
    )

    # Save plot
    plot_filename = filename_prefix
    ax.figure.savefig(plot_filename)
    log.info("Saved plot to file: {}".format(os.path.abspath(plot_filename)))


def load_models(
    include_models: typing.List[str],
    model_dir: os.PathLike = "models",
) -> rsatoolbox.model.Model:
    """Load in model RDMs from HDF5 (.h5 / .hdf5) files.

    :param include_models: list of model names (filename prefixes) to include
    :param model_dir: directory containing model RDMs
    :returns: list of model RDMs
    """
    model_files = pathlib.Path(hydra.utils.to_absolute_path(model_dir)).glob("*.h*5")
    assert model_files, "No hdf5 files found in: {}".format(model_dir)
    rdms = [
        rsatoolbox.rdm.load_rdm(str(model_file))
        for model_file in model_files
        if model_file.stem in include_models
    ]
    assert rdms, "Expected RDM files not found in: {}".format(list(model_files))
    rdms = rsatoolbox.rdm.concat(rdms)
    model = rsatoolbox.model.ModelWeighted("nnls", rdms)

    return model


def fit_bootstrap(
    model: rsatoolbox.model.ModelWeighted,
    data_rdms: rsatoolbox.rdm.RDMs,
    num_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Fit model weights to bootstrap samples of data.

    :param model: model with component RDMs, whose weighted sum will
        approximate the data
    :param data_rdms: RDMs to fit model to
    :param num_bootstrap: number of bootstrap samples of data_rdms to draw
    :return: model weights for each bootstrap sample
    """
    _ = rsatoolbox.util.inference_util.input_check_model(model)
    # Pre-allocate results
    model_weights_evaluations_df = pd.DataFrame(
        data=0,
        index=pd.RangeIndex(num_bootstrap, name="bootstrap sample"),
        columns=pd.CategoricalIndex(
            model.rdm_obj.rdm_descriptors["model_name"], name="model"
        ),
    )

    # Normalize component RDMs to unit-norm,
    # so we can compare their weights fairly.
    model_rdms = model.rdm_obj
    X = sklearn.preprocessing.normalize(model_rdms.dissimilarities).T

    # Fit model weights to bootstrap samples
    for bootstrap_idx in tqdm.trange(num_bootstrap, desc="bootstrap"):
        sampled_rdms, _ = rsatoolbox.inference.bootstrap_sample_rdm(data_rdms)
        y = sampled_rdms.dissimilarities.T

        # Fit weights over bootstrap samples & take mean weights
        model_weights = (
            sklearn.linear_model.Lasso(
                alpha=3e-3,
                fit_intercept=False,
                positive=True,
            )
            .fit(X, y)
            .coef_
        )
        model_weights = model_weights.mean(axis=0)
        model_weights_evaluations_df.iloc[bootstrap_idx] = model_weights

    return model_weights_evaluations_df


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
