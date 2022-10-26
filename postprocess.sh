#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 src/dlio_postprocessor.py --num-proc 4 --output-folder /data/lhovon/dlio_benchmark/output/ --batch-size 8 --epochs 10 --do-eval y

cat /data/lhovon/dlio_benchmark/output/DLIO_report.txt