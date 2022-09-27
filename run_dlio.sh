#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:$(pwd)
echo $PYTHONPATH

num_gpus=${1:-1}

# Run the benchmark with Tensorflow
horovodrun -np $num_gpus python3 src/dlio_benchmark.py --num-files-train 8 --num-samples 313451  --record-length 7 --batch-size 48 --keep-files yes\
    --computation-time 6567 --computation-threads 8 --checkpoint yes --steps-checkpoint 1000 --transfer-size 262144 --read-threads 8    
