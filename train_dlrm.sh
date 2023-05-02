#!/bin/bash
WORKLOAD=${1:-dlrm}
NUM_GPUS=${2:-8}
BATCH_SIZE=${3:-32768}

mpirun -np 1 python3 src/dlio_benchmark.py workload=$WORKLOAD ++workload.reader.batch_size=$BATCH_SIZE ++workload.num_gpus=$NUM_GPUS \
    ++workload.train.total_training_steps=32768 \
    ++workload.evaluation.total_eval_steps=4096 \
    ++workload.evaluation.steps_between_evals=16384 \
    ++workload.checkpoint.steps_between_checkpoints=16384
