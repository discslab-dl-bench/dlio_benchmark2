#!/bin/bash
NUM_GPUS=${1:-8}
BATCH_SIZE=${2:-6}
NUM_STEPS=${3:-2400}
DO_EVAL=${4:-True}
DO_CKPT=${5:-True}

echo $0 $@

mpirun -np $NUM_GPUS python3 src/dlio_benchmark.py workload=bert \
    ++workload.reader.batch_size=$BATCH_SIZE \
    ++workload.num_gpus=$NUM_GPUS \
    ++workload.train.total_training_steps=$NUM_STEPS \
    ++workload.evaluation.total_eval_steps=100 \
    ++workload.workflow.evaluation=$DO_EVAL \
    ++workload.workflow.checkpoint=$DO_CKPT \
    ++workload.checkpoint.checkpoint_after_step=1 \
    ++workload.checkpoint.steps_between_checkpoints=2400
