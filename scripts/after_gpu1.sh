#!/bin/bash
# Start queue c on GPU1 once the earlier AnnoMI DPP run there has written all seeds.
cd "$(dirname "$0")/.."
until [ -f results/policy/annomi/roberta-base_dpp_s44.json ]; do sleep 60; done
exec scripts/queue.sh 1 scripts/jobs_c.txt
