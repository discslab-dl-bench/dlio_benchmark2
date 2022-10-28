#!/bin/bash

# Example script to optionally generate data for the PyTorch framework.
num_procs=${1:-1}

# Generate npz files that we will open with the dataloader
horovodrun -np $num_procs python3 src/dlio_benchmark.py --framework pytorch --do-eval y --data-folder data/ --output-folder output/ --format npz \
    --generate-data yes --generate-only yes --num-files-train 128 --num-files-eval 16 --num-samples 1 --record-length 134217728
