#!/bin/bash

WORKLOAD=${1:-bert}
NUM_GPUS=${2:-1}

mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=$WORKLOAD \
    ++workload.workflow.generate_data=True \
    ++workload.workflow.train=False
