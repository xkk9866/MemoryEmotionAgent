#!/bin/bash
# usage: spire_run.sh <gpu> <dataset> <tag> <seeds...> -- <configs...>
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 CATE_DATA=$PWD/data PYTHONNOUSERSITE=1
PY=${PYTHON_BIN:-python}
g=$1; ds=$2; tag=$3; shift 3
seeds=(); while [ "$1" != "--" ]; do seeds+=($1); shift; done; shift
"$PY" -m cate.spire --dataset $ds --tag $tag --seeds ${seeds[@]} --configs "$@" > logs/spire_${ds}_${tag}_${seeds[0]}.log 2>&1
