#!/bin/bash
WORKLOAD=${1:-dlrm}
NUM_GPUS=${2:-8}
BATCH_SIZE=${3:-2048}

mpirun -np 1 python3 src/dlio_benchmark.py workload=$WORKLOAD ++workload.reader.batch_size=$BATCH_SIZE ++workload.num_gpus=$NUM_GPUS \
    ++workload.train.total_training_steps=500 \
    ++workload.evaluation.total_eval_steps=100 \
    ++workload.evaluation.steps_between_evals=250 \
    ++workload.checkpoint.steps_between_checkpoints=250
