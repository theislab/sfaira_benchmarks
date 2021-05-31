#!/bin/bash

MEMORY="60G"

MODEL_CLASS="CELLTYPE"
ORGANISM=("MOUSE")
ORGANS=("KIDNEY" "PANCREAS")
MODEL_KEYS=("LINEAR")
DEPTH_KEYS=("1")
WIDTH_KEYS=("1")
LEARNING_RATE_KEYS=("1+2")
DROPOUT_RATE_KEYS=("2")
L1_KEYS=("2")
L2_KEYS=("2")

ORGANISATION="THEISLAB"
TOPOLOGY=("0.0.1")
VERSION="0.1"

source  "$(dirname "$0")/base_train_script_cv.sh"
