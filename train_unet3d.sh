#!/bin/bash
NUM_GPUS=${1:-8}
BATCH_SIZE=${2:-4}
NUM_EPOCHS=${3:-50}

mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=unet3d \
    ++workload.reader.batch_size=$BATCH_SIZE \
    ++workload.num_gpus=$NUM_GPUS \
    ++workload.train.epochs=$NUM_EPOCHS \
    ++workload.evaluation.eval_after_epoch=25 \
    ++workload.evaluation.epochs_between_evals=25 \
    ++workload.checkpoint.checkpoint_after_epoch=50
