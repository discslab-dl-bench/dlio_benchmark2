#!/bin/bash

SCRIPT_DIR=$(dirname -- "$( readlink -f -- "$0"; )")

WORKLOAD=${1:-unet3d}
IMAGE=${2:-dlio:latest}
DATA_DIR=${3:-"$SCRIPT_DIR/data"}

docker run -it --rm -v $DATA_DIR:/workspace/dlio/data $IMAGE /bin/bash do_datagen.sh $WORKLOAD
