#!/bin/bash

echo "Please modify docker image name, data and output directories!"

WORKLOAD=${1:-bert}

IMAGE_NAME=
DATA_DIR=

docker run -it --rm -v $DATA_DIR:/workspace/dlio/data $IMAGE_NAME /bin/bash do_datagen.sh $WORKLOAD
