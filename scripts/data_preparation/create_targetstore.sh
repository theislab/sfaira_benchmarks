#!/bin/bash

echo "Creating sfaira target store"

CODE_PATH="/home/icb/${USER}/git"
STORE_PATH="/storage/groups/ml01/workspace/david.fischer/sfaira/store/protein_coding/"
CONFIG_PATH="/storage/groups/ml01/workspace/david.fischer/sfaira/store/configs/"
TARGET_PATH="/storage/groups/ml01/workspace/david.fischer/sfaira/store/targets/"

source "/home/${USER}/.bashrc"
python ${CODE_PATH}/sfaira/sfaira/data/utils_scripts/create_target_universes.py ${STORE_PATH} ${CONFIG_PATH} ${TARGET_PATH}
