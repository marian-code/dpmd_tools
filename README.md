# dpmd-tools

A collection of utilities aimed mainly at manipulating deep MD datasets

Contains tools:
* compare E-V plots for reference data and predictions
* plot data coverage in E-V space
* batch recompute script that does computations on specified remote hosts
* trajectory optimizer to find spacegroup of strcuctures along trajectory
* Oganov fingerprint clustering and selection for datasets
* constrain based filtering of dataset structures

Needs to be installed in appropriate conda environment that has DeePMD-Kit
installed! Use the environment files or install deepmd-kit from conda into
the base environment.