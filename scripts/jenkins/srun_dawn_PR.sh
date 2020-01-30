#!/bin/sh

set -e

BASEPATH_SCRIPT=$(dirname $(realpath -s $0))

export PARALLEL_BUILD_JOBS=24
srun --job-name=dawn_PR \
     --time=00:45:00 \
     --nodes=1 \
     --ntasks-per-node=1 \
     --ntasks-per-core=2 \
     --cpus-per-task=24 \
     --partition=cscsci \
     --constraint=gpu \
     --account=c14 \
     $BASEPATH_SCRIPT/dawn_PR.sh "$@"

