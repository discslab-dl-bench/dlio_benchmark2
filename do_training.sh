#!/bin/bash
WORKLOAD=${1:-bert}
NUM_GPUS=${2:-8}
BATCH_SIZE=

if [[ $WORKLOAD == 'bert' ]]
then
    mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=bert
    mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=bert_eval
else
    mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=$WORKLOAD
fi