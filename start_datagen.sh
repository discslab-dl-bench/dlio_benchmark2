#!/bin/bash

WORKLOAD=${1:-unet3d}

docker run -it --rm -v /raid/data/dlio:/workspace/dlio/data dlio:unet3d-instrumented /bin/bash do_datagen.sh $WORKLOAD
