"""Helper functions for working with RDMs."""

import logging
import os
import pathlib
import typing

import rsatoolbox
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def filename(cfg: DictConfig) -> str:
    """Generate filename for RDM.

    :param cfg: Hydra config object, including task info
    :return: filename for RDM
    """
    return ("sub-{subject}_ses-{session}_rdm").format(
        subject=cfg.array.subject,
        session=cfg.session,
    )


def load_models(
    include_models: typing.List[str],
    model_dir: os.PathLike = "models",
) -> typing.List[rsatoolbox.model.Model]:
    """Load in model RDMs from HDF5 (.h5 / .hdf5) files.

    :param include_models: list of model names (filename prefixes) to include
    :param model_dir: directory containing model RDMs
    :returns: list of model RDMs
    """
    model_list = []
    for model_rdm_file in pathlib.Path(model_dir).glob("*.h*5"):
        rdm = rsatoolbox.rdm.load_rdm(str(model_rdm_file))
        # Assume the model RDMs' conditions are already sorted in the same
        # order as the data RDMs' conditions

        model_name = model_rdm_file.stem
        if model_name in include_models:
            log.debug("Loading model: {}".format(model_name))
            model_list.append(rsatoolbox.model.ModelFixed(model_name, rdm))

    return model_list
