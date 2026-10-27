#!/bin/bash
# Final RLA tables; survives session end. usage: tables.sh <dataset>...
cd "$(dirname "$0")/.."
for d in "$@"; do CUDA_VISIBLE_DEVICES="" python -m cate.rla_report --dataset $d > /d/cate_data/logs/table_$d.txt 2>&1; done
