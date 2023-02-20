#!/bin/bash

IMAGE=${1:-dlio:bert}

docker run -it --rm \
    -v /raid/data/dlio/data:/workspace/dlio/data \
    -v /raid/data/dlio/run_output:/workspace/dlio/hydra_log \
    -v /raid/data/dlio/run_output:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash
