#!/bin/bash

SCRIPT_DIR=$(dirname -- "$( readlink -f -- "$0"; )")

CONTAINER_NAME=${1:-dlio_loic}
IMAGE=${2:-dlio:test}
DATA_DIR=${3:-"$SCRIPT_DIR/data"}
LOGGING_DIR=${4:-"$SCRIPT_DIR/logs"}
OUTPUT_DIR=${5:-"$SCRIPT_DIR/output"}

docker run -it --rm --name $CONTAINER_NAME \
	-v $DATA_DIR:/workspace/dlio/data \
	-v $LOGGING_DIR:/workspace/dlio/hydra_log \
	-v $OUTPUT_DIR:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash
