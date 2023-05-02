#!/bin/bash
WORKLOAD=${1:-unet3d}
NUM_GPUS=${2:-8}
BATCH_SIZE=${3:-4}
NUM_EPOCHS=${4:-50}

mpirun -np $NUM_PROCS python3 src/dlio_benchmark.py workload=$WORKLOAD \
    ++workload.reader.batch_size=$BATCH_SIZE \
    ++workload.num_gpus=$NUM_GPUS \
    ++workload.train.epochs=$NUM_EPOCHS \
    ++workload.evaluation.eval_after_epoch=25 \
    ++workload.evaluation.epochs_between_evals=25 \
    ++workload.checkpoint.checkpoint_after_epoch=50
