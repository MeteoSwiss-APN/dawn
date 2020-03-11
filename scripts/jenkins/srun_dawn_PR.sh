#!/bin/sh

set -e

scriptpath=$(dirname $(realpath -s $0))
rootdir=$scriptpath/../../
image=gtclang/dawn-env-ubuntu19.10

module load daint-gpu
module load sarus

sarus pull $image

if [ -z ${CLANG_GRIDTOOLS_REPOSITORY+x} ]; then
  # ssh port is blocked on compute nodes, go via https port
  export CLANG_GRIDTOOLS_REPOSITORY=git@github.com:MeteoSwiss-APN/clang-gridtools.git
fi
if [ -z ${CLANG_GRIDTOOLS_BRANCH+x} ]; then
  export CLANG_GRIDTOOLS_BRANCH=master
fi

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
     --mount=type=bind,source=$rootdir,destination=/usr/src/dawn \
     $image \
     bash -c " \
     cmake -S /usr/src/dawn -B /usr/src/dawn/build \
          -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DGridTools_DIR=/usr/local/lib/cmake \
          -DPROTOBUF_PYTHON_DIR=/usr/local/lib/python3.7/dist-packages \
          -GNinja && \
     cmake --build /usr/src/dawn/build --parallel 24 --target install && \
     python -m pip install -e /usr/src/dawn/dawn && \
     (cd /usr/src/dawn/build && ctest --progress --output-on-failure) &&
     /usr/src/dawn/scripts/jenkins/build_clang_gridtools.sh -r $CLANG_GRIDTOOLS_REPOSITORY -b $CLANG_GRIDTOOLS_BRANCH"
