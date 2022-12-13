#!/bin/bash

NUM_GPUS=${1:-4}
CONTAINER_NAME=${2:-train_dlio}
WORKLOAD=${3:-bert}
BATCH_SIZE=${4:-4}

docker run -it --rm --name $CONTAINER_NAME -v /raid/data/dlio:/workspace/dlio/data dlio:loic /bin/bash do_training.sh $WORKLOAD $NUM_GPUS $BATCH_SIZE
