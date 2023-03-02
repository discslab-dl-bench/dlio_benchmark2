#!/bin/bash

NUM_GPUS=${1:-2}
CONTAINER_NAME=${2:-dlio_loic}
LOGGING_DIR=${3:-"/raid/data/dlio/run_output"}
IMAGE=${4:-dlio:dlrm-instru}
WORKLOAD=${5:-dlrm}
BATCH_SIZE=${6:-2048}
NUM_STEPS=${7:-1000}

docker run -it --rm --name $CONTAINER_NAME \
	-v /raid/data/dlio/data:/workspace/dlio/data \
	-v $LOGGING_DIR:/workspace/dlio/hydra_log \
	-v /raid/data/dlio/run_output:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash do_training.sh $WORKLOAD $NUM_GPUS $BATCH_SIZE $NUM_STEPS

