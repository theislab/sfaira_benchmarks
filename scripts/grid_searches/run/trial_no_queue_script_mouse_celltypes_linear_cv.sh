#!/bin/bash
if test "${USER}" = "leander.dony"
then
  OUT_PATH_BASE="/storage/groups/ml01/workspace/${USER}/projects/sfaira"
  CODE_PATH="/storage/groups/ml01/code/${USER}"
elif test "${USER}" = "david.fischer"
then
  OUT_PATH_BASE="/storage/groups/ml01/workspace/${USER}/sfaira"
  CODE_PATH=$HOME/git
fi
DATA_PATH="/storage/groups/ml01/workspace/david.fischer/sfaira/store/protein_coding"
CONFIG_PATH="/storage/groups/ml01/workspace/david.fischer/sfaira/store/configs"
TARGET_PATH="/storage/groups/ml01/workspace/david.fischer/sfaira/store/targets"

MODEL_CLASS="CELLTYPE"
ORGANISM="MOUSE"
ORGAN="BLOOD"
MODEL_KEY="LINEAR"
DEPTH_KEY="1"
WIDTH_KEY="1"
LEARNING_RATE_KEY="1"
DROPOUT_RATE_KEY="2"
L1_KEY="2"
L2_KEY="2"

ORGANISATION="THEISLAB"
TOPOLOGY="0.0.1"
VERSION="0.1"

GS_KEY="210427trial_${MODEL_KEY}_${VERSION}_CV"
OUT_PATH="${OUT_PATH_BASE}/grid_searches/${ORGANISM}/${MODEL_CLASS}/${GS_KEY}"

rm -rf "${OUT_PATH}"/jobs
rm -rf "${OUT_PATH}"/logs
rm -rf "${OUT_PATH}"/results
mkdir -p "${OUT_PATH}"/jobs
mkdir -p "${OUT_PATH}"/logs
mkdir -p "${OUT_PATH}"/results

python ${CODE_PATH}/sfaira_benchmarks/scripts/grid_searches/train_script_cv.py $MODEL_CLASS $ORGANISM $ORGAN $MODEL_KEY $DEPTH_KEY $WIDTH_KEY $LEARNING_RATE_KEY $DROPOUT_RATE_KEY $L1_KEY $L2_KEY $ORGANISATION $TOPOLOGY $VERSION $GS_KEY $OUT_PATH/ $DATA_PATH $CONFIG_PATH $TARGET_PATH
