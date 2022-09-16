"""Helper functions for working with RDMs."""

from omegaconf import DictConfig


def filename(cfg: DictConfig) -> str:
    """Generate filename for RDM.

    :param cfg: Hydra config object, including task info
    :return: filename for RDM
    """
    return ("sub-{subject}_ses-{session}_rdm").format(
        subject=cfg.array.subject,
        session=cfg.session,
    )
