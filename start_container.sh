#!/bin/bash

SCRIPT_DIR=$(dirname -- "$( readlink -f -- "$0"; )")

CONTAINER_NAME=${1:-dlio}
IMAGE=${2:-dlio:latest}
LOGGING_DIR=${3:-"$SCRIPT_DIR/logs"}
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/output"

DATA_DIR="/raid/data/dlio/data"
OUTPUT_DIR="/raid/data/dlio/run_output"

docker run -it --rm --name $CONTAINER_NAME \
	-v $DATA_DIR:/workspace/dlio/data \
	-v $LOGGING_DIR:/workspace/dlio/hydra_log \
	-v $OUTPUT_DIR:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash
