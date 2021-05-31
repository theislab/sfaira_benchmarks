# From grid searches to production models with sfaira
The grid searches ran here depend on a conda environment named "sfaira" with sfaira and sfaira_benchmarks installed.
Note that you do not need an R installation at this point anymore.
The grid searches are run in SLURM, if you decide to also do this on a SLURM system, you need to update the SLURM parameters according to your system.

## Prepare data for grid searches
Before running grid searches, prepare the data zoo as described in `scripts/data_preparation/`.

## Run hyperparameter gridsearches
These grid searches can be executed using the scripts in `scripts/grid_searches/run`.
Adjust grid search names in the shell scripts according to your setting.

## Summarize grid searches
The grid searches can be summarised to yield the best hyperparameters per model and anatomic partition via `scripts/grid_searches/final_train_prepare`.
Note: update the grid search names in the executed shell scripts according to the grid searches that you ran.
You can merge grid searches using "+" in a single string.

## Fit production models
Based on the optimal hyperparameters, one can then run final parameter fits via `scripts/grid_searches/final_train_run`, which typically yield models that are uploaded to cloud servers and are then deployed in production.
Update grid search collection names from the previous step.
