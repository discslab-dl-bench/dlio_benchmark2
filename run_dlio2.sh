#!/bin/bash

container_name=${1:-dlrm_dlio2}

# Remove existing container if a previous run was interrupted
if [ "$(docker ps -a | grep $container_name)" ]
then
	docker rm $container_name
fi

# docker run -it --rm --name=$container_name --gpus all -v /raid/data/dlrm_dlio2/dlio2:/workspace/dlio/data/dlrm dlio:latest /bin/bash exec.sh
sudo docker run -it --rm \
	--name=$container_name \
	--gpus all \
	-v /raid/data/dlrm_dlio2/dlio2:/workspace/dlio/data/dlrm \
	-v /raid/data/dlio/run_output:/workspace/dlio/hydra_log \
	-v /raid/data/dlio/run_output:/workspace/dlio/checkpoints \
	dlio:latest /bin/bash exec.sh