#!/bin/bash
WORKLOAD=${1:-bert}
NUM_GPUS=${2:-8}
BATCH_SIZE=${3:-6}

mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=$WORKLOAD \
    ++workload.reader.batch_size=$BATCH_SIZE \
    ++workload.num_gpus=$NUM_GPUS \
    ++workload.train.total_training_steps=2400 \
    ++workload.evaluation.total_eval_steps=100 \
    ++workload.checkpoint.checkpoint_after_step=1 \
    ++workload.checkpoint.steps_between_checkpoints=2400
