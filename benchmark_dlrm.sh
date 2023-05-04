#!/bin/bash
SCRIPT_DIR=$(dirname -- "$( readlink -f -- "$0"; )")

# Change these directories
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/output"

DATA_DIR="/raid/data/dlio/data"
OUTPUT_DIR="/raid/data/dlio/run_output"

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:-dlio}
LOGGING_DIR=${3:-"$SCRIPT_DIR/logs"}
IMAGE=${4:-dlio:latest}

# Get any extra arguments and pass them to the launch script
# The first 2 are number of GPUs and batch size for all workloads 
ARGS=("$@")
PASS=("${ARGS[@]:5}")

docker run -it --rm --name $CONTAINER_NAME \
	-v $DATA_DIR:/workspace/dlio/data \
	-v $LOGGING_DIR:/workspace/dlio/hydra_log \
	-v $OUTPUT_DIR:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash train_dlrm.sh $NUM_GPUS "${PASS[@]}"
