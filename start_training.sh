#!/bin/bash

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:-dlio_loic}
LOGGING_DIR=${3:-"/raid/data/dlio/run_output"}
IMAGE=${4:-dlio:unet3d-instru}
WORKLOAD=${5:-unet3d}
BATCH_SIZE=${6:-4}
NUM_EPOCHS=${7:-50}

docker run -it --rm --name $CONTAINER_NAME \
	-v /raid/data/dlio/data:/workspace/dlio/data \
	-v $LOGGING_DIR:/workspace/dlio/hydra_log \
	-v /raid/data/dlio/run_output:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash do_training.sh $WORKLOAD $NUM_GPUS $BATCH_SIZE $NUM_EPOCHS
