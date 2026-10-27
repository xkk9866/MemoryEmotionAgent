#!/bin/bash
# usage: cf_worker.sh <gpu> <jobfile>; workers share the job file, each line is claimed once (flock).
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 CATE_DATA=$PWD/data HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
PY=${PYTHON_BIN:-python}
JOBS=$2; DONE=$JOBS.claimed; touch $DONE
while true; do
  cmd=$(flock $JOBS.lock bash -c "comm -23 <(grep -v '^$' $JOBS | sort -u) <(sort -u $DONE) | head -1 | tee -a $DONE")
  [ -z "$cmd" ] && break
  tag=$(echo "$cmd" | tr ' ' '\n' | grep -A1 -E -- '--(dataset|tag|seed|fold)' | grep -v -- '--' | tr '\n' '_')
  for try in 1 2; do "$PY" -m $cmd > logs/cf_${tag}.log 2>&1 && break; done
done
echo "worker $1 done $(date)" >> logs/cf_workers.log
