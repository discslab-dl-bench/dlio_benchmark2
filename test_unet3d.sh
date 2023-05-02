#!/bin/bash
NUM_GPUS=${1:-8}
BATCH_SIZE=${2:-4}

mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=unet3d \
    ++workload.reader.batch_size=$BATCH_SIZE \
    ++workload.num_gpus=$NUM_GPUS \
    ++workload.train.epochs=5 \
    ++workload.evaluation.eval_after_epoch=3 \
    ++workload.evaluation.epochs_between_evals=1 \
    ++workload.checkpoint.checkpoint_after_epoch=5

