#!/bin/bash
# usage: when_free.sh <jobfile> [min_free_MiB]; waits until a local GPU has enough free memory, then runs queue.sh on it.
cd "$(dirname "$0")/.."
need=${2:-12000}
while true; do
  gpu=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' -v n=$need '$3-$2>=n {print $1; exit}')
  [ -n "$gpu" ] && exec scripts/queue.sh "$gpu" "$1"
  sleep 120
done
