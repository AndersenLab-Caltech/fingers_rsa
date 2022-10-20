"""Plot RDMs from .hdf5 files."""

import argparse
import pathlib
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import prince
import rsatoolbox
import sklearn.manifold
import xarray as xr

from fingers_rsa import mds_utils


def main() -> None:
    args = parse_args()
    rdm_files = sorted(args.rdm_files)
    rdm_list = [rsatoolbox.rdm.load_rdm(rdm_file) for rdm_file in rdm_files]
    rdms = rsatoolbox.rdm.concat(rdm_list)

    # Plot all RDMs
    fig, _, _ = rsatoolbox.vis.show_rdm(
        rdms,
        cmap=mpl.rcParams["image.cmap"],
        pattern_descriptor=args.pattern_descriptor,
        rdm_descriptor=args.rdm_descriptor,
        show_colorbar="figure",
        vmin=0,
        nanmask=np.zeros((rdms.n_cond, rdms.n_cond), dtype=bool),
    )
    save_path = args.output_dir.joinpath(f"rdm_all_{args.rdm_descriptor}")
    print("Saving RDM plot to:", save_path)
    fig.savefig(save_path)

    # Plot average RDM
    fig, _, _ = rsatoolbox.vis.show_rdm(
        rdms.mean(),
        cmap=mpl.rcParams["image.cmap"],
        pattern_descriptor=args.pattern_descriptor,
        show_colorbar="figure",
        vmin=0,
        nanmask=np.zeros((rdms.n_cond, rdms.n_cond), dtype=bool),
    )
    save_path = args.output_dir.joinpath("rdm_mean")
    print("Saving mean RDM plot to:", save_path)
    fig.savefig(save_path)

    fig, ax = plt.subplots()
    plot_mds(rdms, colors=args.colors, plot_raw=args.plot_raw, ax=ax)
    save_path = args.output_dir.joinpath("mds")
    print("Saving MDS plot to:", save_path)
    fig.savefig(save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rdm-files",
        "-f",
        nargs="+",
        help="RDM files to plot",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        type=pathlib.Path,
        help="Where to save out the plots",
    )
    parser.add_argument(
        "--pattern-descriptor",
        "-p",
        default="finger",
        help="Which pattern descriptor to label RDM axes with",
    )
    parser.add_argument(
        "--rdm-descriptor",
        "-r",
        default="session",
        help="Which RDM descriptor to label RDM axes with",
    )
    parser.add_argument(
        "--colors",
        "-c",
        default=["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677"],
        help="Order of colors to use for pattern-descriptor plotting",
    )
    parser.add_argument(
        "--plot-raw",
        action="store_true",
        default=False,
        help="Whether to overlay individual MDS on the aggregate plot",
    )

    return parser.parse_args()


def plot_mds(
    rdms: rsatoolbox.rdm.RDMs,
    rdm_descriptor: str = "session",
    pattern_descriptor: str = "finger",
    colors: Optional[List[str]] = None,
    scale: bool = False,
    plot_raw: bool = False,
    ax: mpl.axes.Axes = None,
) -> None:
    """Plot multidimensional scaling of representational dissimilarities."""
    # Pre-allocate array for MDS coordinates
    mds_da = xr.DataArray(
        np.nan,
        dims=[rdm_descriptor, pattern_descriptor, "mds_dim"],
        coords={
            rdm_descriptor: rdms.rdm_descriptors[rdm_descriptor],
            pattern_descriptor: rdms.pattern_descriptors[pattern_descriptor],
            "mds_dim": ["mds_1", "mds_2"],
        },
        name="mds",
    )
    rdm_mat = rdms.get_matrices()
    mds = sklearn.manifold.MDS(dissimilarity="precomputed")
    for idx in range(rdms.n_rdm):
        mds_da[{rdm_descriptor: idx}] = mds.fit_transform(rdm_mat[idx])
    assert not mds_da.isnull().any(), "MDS returned NaN for some RDMs"

    # Align MDS coordinates
    mds_da.values = prince.GPA(scale=scale).fit_transform(mds_da.values)

    # Assign some colors to the condition-dimension
    mds_utils.make_mds_plot_prealigned(
        mds_da.values,
        ax=ax,
        condition_order=mds_da.coords[pattern_descriptor].values,
        colors=colors,
    )

    if plot_raw:
        mds_utils.make_mds_plot_prealigned(
            mds_da.values,
            ax=ax,
            condition_order=mds_da.coords[pattern_descriptor].values,
            markers=["."] * len(colors),
            colors=colors,
            raw=True,
        )


if __name__ == "__main__":
    main()
