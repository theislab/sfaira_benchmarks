#!/bin/bash

CODE_PATH=$HOME/git
OUT_PATH_BASE="."
GS_PATH="${OUT_PATH_BASE}grid_searches/"
DATA_PATH="."
CONFIG_PATH="."
TARGET_PATH="."

for GS_KEY in "${GS_KEYS[@]}"; do

  OUT_PATH="${OUT_PATH_BASE}/final_training/${ORGANISM}/${MODEL_CLASS}/${GS_KEY}"
  rm -rf "${OUT_PATH}"/hyperparameter
  mkdir -p "${OUT_PATH}"/hyperparameter

  source "$HOME"/.bashrc
  conda activate sfaira
  python "${CODE_PATH}"/sfaira_benchmarks/scripts/grid_searches/final_train_prepare/write_best_hyperparam.py "${MODEL_CLASS}" "${ORGANISM}" "${ORGANS[*]}" "${GS_KEY}" "${GS_PATH}" "${METRIC}" "${OUT_PATH_BASE}"

done
