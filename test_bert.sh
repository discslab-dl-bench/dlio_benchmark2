#!/bin/bash
NUM_GPUS=${1:-8}
BATCH_SIZE=${2:-6}

mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=bert \
    ++workload.reader.batch_size=$BATCH_SIZE \
    ++workload.num_gpus=$NUM_GPUS \
    ++workload.train.total_training_steps=20 \
    ++workload.evaluation.total_eval_steps=20 \
    ++workload.checkpoint.checkpoint_after_step=1 \
    ++workload.checkpoint.steps_between_checkpoints=19
