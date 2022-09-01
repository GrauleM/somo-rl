#!/bin/bash

#### USAGE
# ./experiments.sh -e singularity_experiment
####

# parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    -e | --exp_name)
        EXPERIMENT_NAME=$2
        ;;
    esac
    shift
done

# runs multiple experiments in parallel
HOME_DIR=/n/holyscratch01/wood_lab/Users/mccarthy
LOG_PATH=$HOME_DIR/rl_work/slurm_logs/${EXPERIMENT_NAME}
EXPERIMENT_DIR=$HOME_DIR/rl_work/somogym-baseline-results/experiments/${EXPERIMENT_NAME}
echo "Submitting runs from experiment ${EXPERIMENT_NAME}..."

NOTE=""
OVERWRITE=False
EXPERIMENT_COUNTER=0

submit_job() {
    RUN_NAME=${2}
    GROUP_NAME=${1}
    RUN_LOG_PATH=$LOG_PATH/${GROUP_NAME}/${RUN_NAME}
    mkdir -p "${RUN_LOG_PATH}"
    echo "Submitting job for group $GROUP_NAME, run $RUN_NAME!"
    sbatch --output=$RUN_LOG_PATH/output.out --error=$RUN_LOG_PATH/error.err --export=HOME_DIR=${HOME_DIR},EXPERIMENT_NAME=${EXPERIMENT_NAME},GROUP_NAME=${GROUP_NAME},RUN_NAME=${RUN_NAME},OVERWRITE=${OVERWRITE},NOTE=${NOTE} ./experimen$
    EXPERIMENT_COUNTER=$(($EXPERIMENT_COUNTER + 1))
}

for group in "$EXPERIMENT_DIR"/*
do
  for run in "$group"/*
   do
     submit_job $(basename $group) $(basename $run)
  done
done

echo "Finished submitting ${EXPERIMENT_COUNTER} runs!"