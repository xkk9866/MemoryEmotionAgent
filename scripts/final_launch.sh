#!/bin/bash
cd ~/research/dpp_esc
pkill -f "cf_worker.sh" ; pkill -f "m cate.spire" ; sleep 2
rm -f results/spire/*/* scripts/jobs_final.txt.claimed
for g in 0 1; do for w in 1 2; do setsid nohup scripts/cf_worker.sh $g scripts/jobs_final.txt > /dev/null 2>&1 < /dev/null & done; done
