# Stability of motor representations after paralysis

[![](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.anaconda.com//)

This repository contains the scripts to generate figures for [Stability of motor representations after paralysis](https://doi.org/10.7554/eLife.74478).

## Dataset
The dataset is available in [NWB:N format](https://www.nwb.org/nwb-neurophysiology/) from https://dandiarchive.org/dandiset/000147.

## Analysis scripts
### Python prerequisites
I recommend installing Python package dependencies using [conda + poetry](https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry).
* Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
* Install [poetry](https://python-poetry.org/docs/#installation)

### Package dependencies
Create a minimal virtual environment:
```bash
conda create -n fingers_rsa python=3.8
conda activate fingers_rsa
```

We use [poetry](https://python-poetry.org/) to specify package dependencies in [pyproject.toml](pyproject.toml) and [poetry.lock](poetry.lock).
To install the dependencies, from the repository directory, run:
```bash
poetry install
```

### Downloading the dataset
To download the dataset, change to the `data` directory and run the download-script:
```bash
cd data/
./download_data.sh
```
For non-Unix environments, you can download the dataset using the equivalent DANDI command:
```bash
dandi download DANDI:000147/0.220913.2243
````

### Running the scripts
We use [Hydra](https://hydra.cc/) to specify configuration files and command-line arguments.
For some scripts, the default configuration is already specified in the script, so you can run the script without any command-line arguments.
```bash
python scripts/peth.py
python scripts/crossval_classify.py
```
These scripts will output figures to an `outputs/date/time/` directory (e.g. `outputs/2022-10-13/11-18-05/`).
The scripts will also log their working/output directory to the console.

To [calculate the representational dissimilarity matrices](scripts/calc_rdm.py) (RDMs), you need to specify the sessions to use for the RSA analysis.
```bash
python scripts/calc_rdm.py --multirun session=2018-09-10,2018-09-17a,2018-09-17b,2018-09-24,2018-09-26,2018-10-01,2018-10-12,2018-10-15,2018-10-17,2018-10-22
```
The following command is equivalent:
```bash
python scripts/calc_rdm.py -m +sweep=all_sessions
```
With the `--multirun` (`-m`) option, Hydra effectively runs `calc_rdm.py` 10 times, once for each session.
Aligning and binning spikes takes some time, so you can parallelize multiruns using `joblib` with the additional command-line argument: `hydra/launcher=joblib`.

The `calc_rdm.py` multirun will generate outputs in `multirun/date/time/run/` directories (e.g., `outputs/2022-10-13/09-51-04/0`).
The output RDM absolute paths are also logged by `calc_rdm.py`, e.g.:

```
[2022-10-13 09:51:27,590][__main__][INFO] - Saving RDMs to: /home/user/Development/fingers_rsa/multirun/2022-10-13/09-51-04/session=2018-09-24/sub-P1_ses-2018-09-24_rdm.hdf5
```

The RDM files (absolute-path) are used as inputs to [representational similarity analysis](scripts/representational_similarity_analysis.py). For the example above, you would run:
```bash
python scripts/representational_similarity_analysis.py rdm_files="/home/user/Development/fingers_rsa/multirun/2022-10-13/09-51-04/*/*_rdm.hdf5"
````

For [representational dynamics analysis](scripts/representational_dynamics_analysis.py) (RDA), RDMs need to be calculated for multiple sessions and times. You can run:
```bash
python scripts/calc_rdm.py -m +sweep=all_sessions_and_times
```
This will take ~10 minutes even with parallelization.

Then, similar to RSA, you would pass in the resulting output RDM files to the RDA script:
```bash
python scripts/representational_dynamics_analysis.py rdm_files="/home/user/Development/fingers_rsa/multirun/date/time/*/*_rdm.hdf5"
```

### Development environment
I developed and tested these Python scripts on [Ubuntu 20.04 LTS](https://releases.ubuntu.com/focal/).
While they should work on other operating systems, I have not tested them.
If you run into any issues, please open an issue and list your operating system and Python version.


## Citation
<div class="csl-entry">Charles Guan, Tyson Aflalo, Carey Y Zhang, Elena Amoruso, Emily R Rosario, Nader Pouratian, Richard A Andersen (2022) <b>Stability of motor representations after paralysis</b> <i>eLife</i> <b>11</b>:e74478. https://doi.org/10.7554/eLife.74478</div>
