#!/bin/bash
cd "$(dirname "$0")/.."
for d in annomi p4g esconv; do for f in roberta-large_none roberta-base_none; do
  CUDA_VISIBLE_DEVICES="" python -m cate.rla_ablation --dataset $d --prefix $f 2>&1 | grep -v -i warn >> /d/cate_data/logs/ablations.txt
done; done
echo ALLDONE >> /d/cate_data/logs/ablations.txt
