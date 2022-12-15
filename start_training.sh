#!/bin/bash

echo "Please modify docker image name, data and output directories!"

NUM_GPUS=${1:-4}
CONTAINER_NAME=${2:-train_dlio}
BATCH_SIZE=${3:-4}
WORKLOAD=${4:-bert}

IMAGE_NAME=
OUTPUT_DIR=
DATA_DIR=


docker run -it --rm --name $CONTAINER_NAME \
    -v $OUTPUT_DIR:/workspace/dlio/hydra_log \
    -v $DATA_DIR:/workspace/dlio/data \
    $IMAGE_NAME /bin/bash do_training.sh $WORKLOAD $NUM_GPUS $BATCH_SIZE
