#!/bin/bash

IMAGE=${1:-dlio:dlrm}

docker run -it --rm \
    -v /raid/data/dlrm_dlio2/dlio2:/workspace/dlio/data/dlrm \
    -v /raid/data/dlio/run_output:/workspace/dlio/hydra_log \
    -v /raid/data/dlio/run_output:/workspace/dlio/checkpoints \
    $IMAGE /bin/bash
