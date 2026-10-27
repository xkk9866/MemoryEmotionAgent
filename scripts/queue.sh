#!/bin/bash
# usage: queue.sh <gpu> <jobfile>; each line of jobfile is a python -m command, retried up to 3 times.
export HF_HOME='D:\hf_cache' CUDA_VISIBLE_DEVICES=$1
cd "$(dirname "$0")/.."
log=/d/cate_data/logs/queue_$(basename "$2" .txt).log
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  for try in 1 2 3; do
    echo "== $(date +%T) try$try $cmd" >> "$log"
    python -m $cmd >> "$log" 2>&1 && break
  done
done < "$2"
echo "== QUEUE DONE" >> "$log"
