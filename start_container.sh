#!/bin/bash

CONTAINER_NAME=${1:-dlio_loic}
IMAGE=${2:-dlio:test}
LOGGING_DIR=${3:-"/raid/data/dlio/run_output"}

docker run -it --rm --name $CONTAINER_NAME \
	-v  /dl-bench/lhovon/dlio/data:/workspace/dlio/data \
	-v $LOGGING_DIR:/workspace/dlio/hydra_log \
	-v /raid/data/dlio/run_output:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash
