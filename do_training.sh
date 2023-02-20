#!/bin/bash
WORKLOAD=${1:-bert}
NUM_GPUS=${2:-8}
BATCH_SIZE=

mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=$WORKLOAD