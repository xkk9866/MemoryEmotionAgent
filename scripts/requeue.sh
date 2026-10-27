#!/bin/bash
# usage: requeue.sh <jobfile> <gpu:workers>...  re-claims done/running jobs only, then starts workers
cd ~/research/dpp_esc
J=$1; shift
pkill -f "cf_worker.sh $" 2>/dev/null; pkill -f "cf_worker.sh [0-9] $J" ; sleep 1
: > $J.claimed
while IFS= read -r cmd; do
  ds=$(echo "$cmd" | sed -E 's/.*--dataset ([a-z0-9]+).*/\1/'); tag=$(echo "$cmd" | sed -E 's/.*--tag ([a-z0-9-]+).*/\1/'); s=$(echo "$cmd" | sed -E 's/.*--seeds ([0-9]+).*/\1/')
  if [ -f results/spire/$ds/${tag}_s$s.json ] || pgrep -f -- "--dataset $ds --tag $tag --seeds $s " > /dev/null; then echo "$cmd" >> $J.claimed; fi
done < $J
for gw in "$@"; do g=${gw%%:*}; n=${gw##*:}; for i in $(seq $n); do setsid nohup scripts/cf_worker.sh $g $J > /dev/null 2>&1 < /dev/null & done; done
