#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:$(pwd)
echo $PYTHONPATH

num_gpus=${1:-1}

# Run the benchmark with Tensorflow
horovodrun -np $num_gpus python3 src/dlio_benchmark.py --num-files-train 500 --num-samples 313451  --record-length 2500 --batch-size 48 --keep-files yes\
    --computation-time 0.968 --computation-threads 8 --checkpoint yes --steps-checkpoint 6250 --transfer-size 262144 --read-threads 8 --data-folder /workspace/dlio/data 
