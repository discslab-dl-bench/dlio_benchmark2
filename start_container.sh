#!/bin/bash

IMAGE=${1:-dlio:dlrm}

SCRIPT_DIR=$(dirname -- "$( readlink -f -- "$0"; )")
LOGGING_DIR="${SCRIPT_DIR}/output"

mkdir -p $LOGGING_DIR

docker run -it --rm --name dlrm_loic \
	-v /raid/data/dlio/data:/workspace/dlio/data \
	-v $LOGGING_DIR:/workspace/dlio/hydra_log \
	-v /raid/data/dlio/run_output:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash
