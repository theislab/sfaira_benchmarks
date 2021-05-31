#!/bin/bash

echo "Creating sfaira data store"

CODE_PATH="/home/icb/${USER}/git"
OUT_PATH="/storage/groups/ml01/workspace/david.fischer/sfaira/store"
DATA_PATH="/storage/groups/ml01/datasets/projects/20200101_Various_SfairaDataRepository_leander.dony/raw/"
META_PATH="${OUT_PATH}/meta/"
CACHE_PATH="${OUT_PATH}/cache/"
STORE_PATH="${OUT_PATH}/protein_coding/"
STORE_TYPE="H5AD"

source "/home/${USER}/.bashrc"
python ${CODE_PATH}/sfaira/sfaira/data/utils_scripts/write_store.py ${DATA_PATH} ${META_PATH} ${CACHE_PATH} ${STORE_PATH} ${STORE_TYPE}
