"""Plot RDMs from .hdf5 files."""

import argparse
import pathlib

import matplotlib as mpl
import numpy as np
import rsatoolbox


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
    fig.savefig(args.output_dir.joinpath(f"rdm_all_{args.rdm_descriptor}"))

    # Plot average RDM
    fig, _, _ = rsatoolbox.vis.show_rdm(
        rdms.mean(),
        cmap=mpl.rcParams["image.cmap"],
        pattern_descriptor=args.pattern_descriptor,
        show_colorbar="figure",
        vmin=0,
        nanmask=np.zeros((rdms.n_cond, rdms.n_cond), dtype=bool),
    )
    fig.savefig(args.output_dir.joinpath("rdm_mean"))


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

    return parser.parse_args()


if __name__ == "__main__":
    main()
