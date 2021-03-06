#!/bin/bash

MEMORY="150G"

MODEL_CLASS="EMBEDDING"
ORGANISM=("HUMAN")
ORGANS=("ADIPOSETISSUE" "ADRENALGLAND" "ARTERY" "BLOOD" "BONEMARROW" "BRAIN" "CHORIONICVILLUS" "ESOPHAGUS" "EYE" "GALLBLADDER" "HEART" "INTESTINE" "KIDNEY" "LIVER" "LUNG" "MUSCLEORGAN" "OVARY" "PANCREAS" "PLACENTA" "PROSTATEGLAND" "RIB" "SKELETON" "SKINOFBODY" "SPINALCORD" "SPLEEN" "STOMACH" "TESTIS" "THYMUS" "THYROIDGLAND" "TRACHEA" "URETER" "URINARYBLADDER" "UTERUS" "VAULTOFSKULL")
MODEL_KEYS=("PCA")

ORGANISATION="THEISLAB"
VERSION="0.1"

source  "$(dirname "$0")/base_train_script_pca_cv.sh"
