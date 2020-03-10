#!/bin/sh

set -e

BASEPATH_SCRIPT=$(dirname $(realpath -s $0))

module load daint-gpu
module load sarus

echo $BASEPATH_SCRIPT
rootdir=$BASEPATH_SCRIPT/../../

image=gtclang/dawn-env-ubuntu19.04

sarus pull $image

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
     sarus run \
     --mount=type=bind,source=$SCRATCH,destination=$SCRATCH \
     $image \
     cmake -S $rootdir -B $rootdir/build \
    -DBUILD_TESTING=ON \
    -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DGridTools_DIR=/usr/local/lib/cmake \
    -DPROTOBUF_PYTHON_DIR=/usr/local/lib/python3.7/dist-packages \
    -GNinja

#     $BASEPATH_SCRIPT/dawn_PR.sh "$@"
