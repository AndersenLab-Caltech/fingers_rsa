"""Helper functions for working with RDMs."""

from omegaconf import DictConfig


def filename(cfg: DictConfig) -> str:
    return (
        "rdm_{task}_{subject}_arr{array}{neurons}_{distance_metric}" "_{time_window}"
    ).format(
        task=cfg.task.name,
        subject=cfg.array.subject,
        array=cfg.array.index,
        neurons=(("_" + cfg.neurons.name) if ("neurons" in cfg) else ""),
        distance_metric=cfg.metrics.distance,
        time_window=[cfg.window.start, cfg.window.start + cfg.window.length],
    )
