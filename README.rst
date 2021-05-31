Sfaira benchmarks
=====================================================================

Collection of scripts and notebooks to create panels in publication.
This repository contains grid search preparation, execution and evaluation code used in the original sfaira_ publication_.

Next to python and shell scripts for grid searches and jupyter notebooks for results evaluation, this repository contains shallow infrastructure for defining hyperparameters in grid searches under `sfaira_benchmarks/`. 
Install this package via `pip install -e .` into a python environment with an existing sfaira installation to make this infrastructure available to the grid search scripts defined in this repository.

Before running grid searches, prepare the data zoo as described in `scripts/data_preparation/`.
Grid searches and production model training can be run using the scripts as described in `scripts/grid_searches/`.


.. _sfaira: https://sfaira.readthedocs.io
.. _publication : https://www.biorxiv.org/content/10.1101/2020.12.16.419036v1
