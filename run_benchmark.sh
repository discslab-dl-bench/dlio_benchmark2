#!/bin/bash

usage() {
	echo -e "Usage: $0 [OPTIONS] -- [EXTRA ARGS]"
	echo -e "Convenience script to launch the DLIO benchmark and container."
	echo -e "The given run-script (-rs) will be launched within the container, after flushing the caches and start iostat to collect disk and cpu usage statistics."
	echo -e "If no script is given, an interactive session to the container will be started."
	echo -e "An optional data-generation script can be passed and will be used first."
	echo -e "Optionally generating the data beforehand. and runs the given script or an interactive session if none is given."
	echo -e "\nOptions:"
	echo -e "  -h, --help\t\t\tPrint this message."
	echo -e "  -dd, --data-dir\t\tDirectory where the training data is read and generated."
	echo -e "  -od, --output-dir\t\tOutput directory for log and checkpoint files."
	echo -e "  -bd, --disk\t\t\tDisk to trace using iostat. Can be passed multiple times with the different disks to trace."
	echo -e "  -i, --image-name\t\tName of the docker image to launch the container from. Defaults to 'dlio:latest'."
	echo -e "  -c, --container-name\t\tName to give the docker container. Defaults to none."
	echo -e "  -gs, --datagen-script\t\tScript to generate the data for this run. If empty, data will be assumed to exist in data-dir."
	echo -e "  -rs, --run-script\t\t\tScript to launch within the container. Defaults to launching bash within the container."
	echo -e "\nExtra args:"
	echo -e "  Any extra arguments passed after after '--' will be passed as is to the DLIO launch script."
	echo ""
	exit 1
}

main() {
	SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
	
	if [ "${EUID:-$(id -u)}" -ne 0 ]
	then
		echo "Run script as root"
		usage
		exit 1
	fi

	# Defaults
	DATA_DIR=${SCRIPT_DIR}/data
	OUTPUT_DIR=${SCRIPT_DIR}/output
	IMAGE_NAME="dlio:latest"
	CONTAINER_NAME="dlio"
	DATAGEN_SCRIPT=
	RUN_SCRIPT=
	BLKDEVS=()

	while [ $# -gt 0 ]; do
	case "$1" in
		-h | --help ) usage ;;
		-od | --output-dir ) OUTPUT_DIR="$2"; shift 2 ;;
		-dd | --data-dir ) DATA_DIR="$2"; shift 2 ;;
		-bd | --blk-dev ) BLKDEVS+="$2 "; shift 2 ;;
		-i | --image-name ) IMAGE_NAME="$2"; shift 2 ;;
		-c | --container-name ) CONTAINER_NAME="$2"; shift 2 ;;
		-gs | --datagen-script ) DATAGEN_SCRIPT="$2"; shift 2 ;;
		-rs | --run-script ) RUN_SCRIPT="$2"; shift 2 ;;
		-- ) shift; break ;;
		* ) echo "Invalid option"; usage ;;
	esac
	done

	EXTRA_ARGS=$@

	mkdir -p $DATA_DIR
	mkdir -p $OUTPUT_DIR

	# Remove existing container from a previous run (docker won't let you use the same name otherwise).
	if [ ! -z $CONTAINER_NAME ]
	then
		if [ $(docker ps -a --format "{{.Names}}" | grep "^${CONTAINER_NAME}$") ]
		then
			echo "Container name already used by an existing container. Attempting to remove it."
			# Check if the name conflict comes from a running container, in which case we have to kill it first.
			if [ $(docker ps --format "{{.Names}}" | grep "^${CONTAINER_NAME}$") ]
			then
				docker kill $CONTAINER_NAME
				[[ $? != 0 ]] && exit -1
				docker rm $CONTAINER_NAME
			else
				# We can now remove the container
				docker rm $CONTAINER_NAME
				[[ $? != 0 ]] && exit -1
			fi
			echo "Successfully removed container."
		fi
		CONTAINER_NAME_ARG="--name $CONTAINER_NAME"
	fi

	# We could launch the container, copy scripts from host to container and lauch them
	# that way. This would remove the need to rebuild the image every time the scripts change
	
	# Launch the container in the background
	docker run --ipc=host -it -d --rm $CONTAINER_NAME_ARG \
			-v $DATA_DIR:/workspace/dlio/data \
			-v $OUTPUT_DIR:/workspace/dlio/output \
			$IMAGE_NAME /bin/bash

	# Launch the data generation script within the container
	if [ ! -z $DATAGEN_SCRIPT ]
	then
		docker cp $DATAGEN_SCRIPT $CONTAINER_NAME:/workspace/dlio
		docker exec $CONTAINER_NAME /bin/bash $DATAGEN_SCRIPT
	fi

	# Drop caches on the host
	sync && echo 3 > /proc/sys/vm/drop_caches

	# Launch DLIO within the container
	if [ ! -z $RUN_SCRIPT ]
	then
		# Start iostat here instead of within DLIO which could become problematic in multi-instance runs
		iostat -mdxtcy -o JSON $BLKDEVS 1 > $OUTPUT_DIR/iostat.json &
		IOSTAT_PID=$!

		docker cp $RUN_SCRIPT $CONTAINER_NAME:/workspace/dlio
		docker exec $CONTAINER_NAME /bin/bash $RUN_SCRIPT $EXTRA_ARGS

		kill -SIGINT $IOSTAT_PID
	else
		docker exec -it $CONTAINER_NAME /bin/bash
	fi

	# Container is left running if you want to connect to it and check something
}

main $@
