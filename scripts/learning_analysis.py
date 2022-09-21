"""Analyze how representational structure changes over sessions."""

import glob
import logging
import os

import hydra
import numpy as np
import pandas as pd
import rsatoolbox
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

from fingers_rsa import rdm_utils

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="learning", version_base=None)
def main(cfg: DictConfig) -> None:
    log.debug("Config args:\n{}".format(OmegaConf.to_yaml(cfg)))

    # Convert config parameters as needed
    data_rdm_files = glob.glob(cfg.rdm_files)
    filename_prefix = filename(cfg)

    # Load in models. These should be saved as RDM files, too.
    if cfg.metrics.similarity.startswith("cosine"):
        cfg.rsa.models += ["unstructured"]
    model_rdms = rdm_utils.load_models(cfg.rsa.models)

    # Read in the RDM files
    data_rdm_list = [rsatoolbox.rdm.load_rdm(rdm_file) for rdm_file in data_rdm_files]
    # Assume that session uses year-month-day[a/b] format,
    # so we can sort alphabetically
    data_rdm_list.sort(key=lambda rdm: rdm.rdm_descriptors["session"][0])
    data_rdms = rsatoolbox.rdm.concat(data_rdm_list)
    session_order = data_rdms.rdm_descriptors["session"]

    # Validate data
    np.testing.assert_array_equal(
        data_rdms.rdm_descriptors["window_length"],
        data_rdms.rdm_descriptors["window_length"][0],
        err_msg="Window lengths should be the same for all RDMs",
    )
    assert data_rdms.n_rdm > 1, "Need more than one RDM to calculate trends"

    results: rsatoolbox.inference.Result = rsatoolbox.inference.eval_fixed(
        models=model_rdms, data=data_rdms, method=cfg.metrics.similarity
    )

    # Save results to hdf5
    results_filename = filename_prefix + ".hdf5"
    results.save(results_filename, file_type="hdf5", overwrite=True)
    log.info(
        "Saved rsatoolbox results to file: {}".format(os.path.abspath(results_filename))
    )

    # Convert results to DataFrame for plotting
    results_df = pd.DataFrame(
        # Evaluations includes an initial singleton dimension
        results.evaluations.squeeze(axis=0),
        index=pd.Index(data=[model.name for model in results.models], name="model"),
        columns=data_rdms.rdm_descriptors["session"],
    )
    results_df = results_df.melt(
        value_name=cfg.metrics.similarity, var_name="session", ignore_index=False
    ).reset_index()
    results_df["session_index"] = results_df.session.apply(session_order.index)

    # Plot results across sessions
    g = sns.lmplot(
        data=results_df,
        x="session_index",
        y=cfg.metrics.similarity,
        hue="model",
        markers=["o", "x"],
    )
    g.fig.savefig(filename_prefix)
    log.info("Saved lmplot to file: {}".format(os.path.abspath(filename_prefix)))


def filename(cfg: DictConfig) -> str:
    """Generate filename for analysis output files.

    :param cfg: Hydra config object, including task info
    :return: filename for learning analysis
    """
    return ("sub-{subject}_learning").format(
        subject=cfg.array.subject,
    )


if __name__ == "__main__":
    main()
