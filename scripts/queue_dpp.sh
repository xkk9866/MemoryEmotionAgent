#!/bin/bash
# usage: queue_dpp.sh <gpu> <prior>
export HF_HOME='D:\hf_cache' CUDA_VISIBLE_DEVICES=$1
cd "$(dirname "$0")/.."
for ds in esconv annomi; do
  ep=8; [ $ds = annomi ] && ep=10
  python -m cate.policy_train --dataset $ds --prior $2 --seeds 42 43 44 --epochs $ep >> /d/cate_data/logs/q_${ds}_$2.log 2>&1
done
