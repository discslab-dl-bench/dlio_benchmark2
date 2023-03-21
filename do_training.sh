#!/bin/bash
WORKLOAD=${1:-dlrm}
NUM_GPUS=${2:-8}
BATCH_SIZE=${3:-2048}
NUM_STEPS=${4:-1000}

mpirun -np 1 python3 src/dlio_benchmark.py workload=$WORKLOAD \
    ++workload.reader.batch_size=$BATCH_SIZE \
    ++workload.num_gpus=$NUM_GPUS \
    ++workload.train.total_training_steps=$NUM_STEPS
