#!/bin/bash
WORKLOAD=${1:-bert}
NUM_GPUS=${2:-4}
BATCH_SIZE=

mpirun -np 8 $NUM_GPUS src/dlio_benchmark.py workload=$WORKLOAD ++workload.workflow.profiling=True ++workload.profiling.profiler=iostat ++workload.profiling.iostat_devices=[sda,sdb]