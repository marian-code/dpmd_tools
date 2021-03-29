[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

# dpmd-tools

A collection of utilities aimed mainly at manipulating deep MD datasets

Contains tools:
* compare E-V plots for reference data and predictions
* plot data coverage in E-V space
* batch recompute script that does computations on specified remote hosts
* trajectory optimizer to find spacegroup of strcuctures along trajectory
* Oganov fingerprint clustering and selection for datasets
* constrain based filtering of dataset structures - supports iterative dataset
  growing
* upload_script that copies dataset files from local to remote server and neatly
  organizes the directories

# Environment

Needs to be installed in appropriate conda environment that has DeePMD-Kit
installed if you want to use dataset filtering based on NN predictions or compare
graphs! Use the environment files or install deepmd-kit from conda into your
the base environment.

```bash
conda env create -f environment_cpu.yml
conda activade dpmd_cpu
```

# Install

The package must be installed in order for the submodules to be importable.
Otherwise some scripts will not work.

```bash
git clone https://github.com/ftl-fmfi/dpmd_tools.git
cd dpmd_tools
pip install -e .  # -e is for editable mode
```

If you want to read data from LAMMPS MD trajectories
```
pip install git+https://github.com/ftl-fmfi/py-extract-frames@master#egg=py-extract-frames
```

# Example

100 clusters found in ge136 MetaD trajectory by `data_cluster.py`. For 300K
136 atomstructures, fingerprinting takes roughly 2 hours and clustering is a
matter of minutes.

![Alt Text](data/clusters.png)