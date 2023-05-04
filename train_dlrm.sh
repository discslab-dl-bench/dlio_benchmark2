#!/bin/bash
NUM_GPUS=${1:-8}
BATCH_SIZE=${2:-32768}
NUM_STEPS=${3:-32768}

mpirun -np 1 python3 src/dlio_benchmark.py workload=dlrm ++workload.num_gpus=$NUM_GPUS \
    ++workload.reader.batch_size=$BATCH_SIZE \
    ++workload.train.total_training_steps=$NUM_STEPS \
    ++workload.evaluation.total_eval_steps=2048 \
    ++workload.evaluation.steps_between_evals=16384 \
    ++workload.checkpoint.steps_between_checkpoints=16384
