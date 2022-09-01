#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition=shared
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=thomasmccarthy@college.harvard.edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16000
#SBATCH -t 0-12:00
# ====
# you must set the following variables using the --export flag when running this slurm script. view experiments.sh for an example.
# ====
# HOME_DIR
# EXPERIMENT_NAME
# GROUP_NAME
# RUN_NAME
# OVERWRITE
# NOTE
# ====
LOG_FILE=$HOME_DIR/rl_work/slurm_logs/${EXPERIMENT_NAME}/${GROUP_NAME}/${RUN_NAME}/stdout.txt
if [ "$OVERWRITE" = "True" ]; then OVERWRITE_ARG=" -o"; else OVERWRITE_ARG=""; fi
if [ "$NOTE" != "" ]; then NOTE_ARG=" --note '${NOTE}'"; else NOTE_ARG=""; fi
echo "========================="
echo "Starting training session"
echo "========================="
read -r -d '' EXEC_TRAINING << EOM
mkdir -p "${LOG_FILE%/*}" && touch "$LOG_FILE"
cd /rl_work/somo-rl/somo_rl
script -c "python3.7 train_benchmark_policy.py -e ${EXPERIMENT_NAME} -g ${GROUP_NAME} -r ${RUN_NAME}${OVERWRITE_ARG}${NOTE_ARG}" $LOG_FILE
EOM

echo -e "Executing the following script ...\n$EXEC_TRAINING"
singularity exec -e --bind $HOME_DIR/rl_work:/rl_work somo_rl.img /bin/sh -c "$EXEC_TRAINING"

echo "Done! We hope you enjoyed the experiment."