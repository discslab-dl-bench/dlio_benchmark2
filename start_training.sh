#!/bin/bash

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:-dlio_loic}
WORKLOAD=${3:-dlrm}
BATCH_SIZE=${4:-2048}

docker run -it --rm --name $CONTAINER_NAME \
	-v /raid/data/dlrm_dlio2/dlio2:/workspace/dlio/data/dlrm \
	-v /raid/data/dlio/run_output:/workspace/dlio/hydra_log \
	-v /raid/data/dlio/run_output:/workspace/dlio/checkpoints \
    dlio:$WORKLOAD /bin/bash do_training.sh $WORKLOAD $NUM_GPUS $BATCH_SIZE
