#!/bin/bash

CODE_PATH=$HOME/git
OUT_PATH_BASE="."
GS_PATH="${OUT_PATH_BASE}/grid_searches/"
DATA_PATH="."
CONFIG_PATH="."
TARGET_PATH="."

SBATCH_PART_C=""
SBATCH_PART_G=""
SBATCH_GRES_C=""
SBATCH_GRES_G=""
SBATCH_QOS_C=""
SBATCH_QOS_G=""
SBATCH_MEM=$MEMORY
SBATCH_C=""
SBATCH_NICE=""
SBATCH_TIME=""
SBATCH_EXCLUDE_G=""
SBATCH_EXCLUDE_C=""
SBATCH_P=""
SBATCH_GRES=""
SBATCH_QOS=""
SBATCH_EXCLUDE=""

for GS_KEY in "${GS_KEYS[@]}"; do

OUT_PATH="${OUT_PATH_BASE}/final_training/${ORGANISM}/${MODEL_CLASS}/${GS_KEY}/"

rm -rf "${OUT_PATH}"/jobs
rm -rf "${OUT_PATH}"/logs
rm -rf "${OUT_PATH}"/results
mkdir -p "${OUT_PATH}"/jobs
mkdir -p "${OUT_PATH}"/logs
mkdir -p "${OUT_PATH}"/results

for file_with_path in $OUT_PATH/hyperparameter/*; do
  best_param_file=$(basename -- "$file_with_path")
  sleep 0.02
  job_file="${OUT_PATH}/jobs/run_${best_param_file%.*}.cmd"
  echo "#!/bin/bash
#SBATCH -J ${MODEL_CLASS}_${a}_${o}_${m}_${d}_${w}_${lr}_${dr}_${l1}_${l2}
#SBATCH -o ${OUT_PATH}/jobs/run_${best_param_file%.*}.out
#SBATCH -e ${OUT_PATH}/jobs/run_${best_param_file%.*}.err
#SBATCH -p ${SBATCH_P}
#SBATCH -q ${SBATCH_QOS}
#SBATCH --gres=${SBATCH_GRES}
#SBATCH -t ${SBATCH_TIME}
#SBATCH --mem=${SBATCH_MEM}
#SBATCH -c ${SBATCH_C}
#SBATCH --nice=${SBATCH_NICE}
#SBATCH --exclude=${SBATCH_EXCLUDE}

source $HOME/.bashrc
conda activate sfaira

python ${CODE_PATH}/sfaira_benchmarks/scripts/grid_searches/final_train_run/train_script_final_model.py ${MODEL_CLASS} $OUT_PATH $DATA_PATH $CONFIG_PATH $TARGET_PATH $best_param_file
" > "${job_file}"
  sbatch "$job_file"
done

done
