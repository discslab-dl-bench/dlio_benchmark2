#!/bin/bash
SCRIPT_DIR=$(dirname -- "$( readlink -f -- "$0"; )")

WORKLOAD=${1:-unet3d}
NUM_GPUS=${2:-8}
CONTAINER_NAME=${3:-dlio_loic}
IMAGE=${4:-dlio:test}
DATA_DIR=${5:-"$SCRIPT_DIR/data"}
LOGGING_DIR=${6:-"$SCRIPT_DIR/logs"}
OUTPUT_DIR=${7:-"$SCRIPT_DIR/output"}

docker run -it --rm --name $CONTAINER_NAME \
	-v $DATA_DIR:/workspace/dlio/data \
	-v $LOGGING_DIR:/workspace/dlio/hydra_log \
	-v $OUTPUT_DIR:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash train_$WORKLOAD.sh
