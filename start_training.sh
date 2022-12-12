#!/bin/bash
WORKLOAD=${1:-bert}

docker run -it --rm -v /raid/data/dlio:/workspace/dlio/data dlio:loic /bin/bash training.sh $WORKLOAD
