#!/bin/bash

CODE_PATH="/home/icb/${USER}/git"
SFAIRA_PATH="${CODE_PATH}/sfaira/"

source "/home/${USER}/.bashrc"

mkdir -p "${HOME}/R/x86_64-pc-linux-gnu-library/4.0/"
Rscript -e "install.packages(c('Seurat'), repos='https://ftp.gwdg.de/pub/misc/cran/', lib='$HOME/R/x86_64-pc-linux-gnu-library/4.0/')"
pip install -U tensorflow

pip uninstall -y sfaira
pip uninstall -y sfaira_extension
# install sfaira
pip install -e "${SFAIRA_PATH}"
