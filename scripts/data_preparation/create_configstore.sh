#!/bin/bash

echo "Creating sfaira config store"

CODE_PATH="/home/icb/${USER}/git"
STORE_PATH="/storage/groups/ml01/workspace/david.fischer/sfaira/store/protein_coding/"
CONFIG_PATH="/storage/groups/ml01/workspace/david.fischer/sfaira/store/configs/"

source "/home/${USER}/.bashrc"
python ${CODE_PATH}/sfaira/sfaira/data/utils_scripts/create_anatomical_configs_store.py ${STORE_PATH} ${CONFIG_PATH}
