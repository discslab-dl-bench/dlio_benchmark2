#!/bin/bash

NUM_GPUS=${1:-bert}
CONTAINER_NAME=${2:-bert}
WORKLOAD=${3:-bert}

docker run -it --rm --name $CONTAINER_NAME \
	-v /raid/data/dlrm_dlio2/dlio2:/workspace/dlio/data/dlrm \
	-v /raid/data/dlio/run_output:/workspace/dlio/hydra_log \
	-v /raid/data/dlio/run_output:/workspace/dlio/checkpoints \
    dlio:$WORKLOAD /bin/bash run_dlrm.sh
