#!/bin/bash

WORKLOAD=${1:-bert}
NUM_GPUS=${2:-4}
BATCH_SIZE=${3:-6}

mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=$WORKLOAD ++workload.reader.batch_size=$BATCH_SIZE ++workload.workflow.profiling=True ++workload.profiling.profiler=iostat ++workload.profiling.iostat_devices=[sda,sdb]