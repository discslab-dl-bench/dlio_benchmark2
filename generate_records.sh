#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:$(pwd)
echo $PYTHONPATH

num_gpus=8
num_files=8

if [ $# -eq 1 ]
then
	num_gpus=$1
fi

if [ $# -eq 2 ]
then
	num_gpus=$1
	num_files=$2
fi

horovodrun -np $num_gpus python3 src/dlio_benchmark.py --record-length 7 --num-samples 313451 --generate-data yes --generate-only yes --num-files-train $num_files --file-prefix "part" 
