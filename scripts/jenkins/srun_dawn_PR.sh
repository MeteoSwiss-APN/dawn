#!/bin/sh

scriptpath=$(dirname `which $0`)
rootdir=$scriptpath/../../
image=gtclang/dawn-env-ubuntu19.10

module load daint-gpu
module load sarus

# Clone clang-gridtools here because the image does not have the ssh key
if [ -z ${CLANG_GRIDTOOLS_REPOSITORY+x} ]; then
  # ssh port is blocked on compute nodes, go via https port
  export CLANG_GRIDTOOLS_REPOSITORY=ssh://git@ssh.github.com:443/MeteoSwiss-APN/clang-gridtools.git
fi
if [ -z ${CLANG_GRIDTOOLS_BRANCH+x} ]; then
  export CLANG_GRIDTOOLS_BRANCH=master
fi

git clone --depth 1 -b $CLANG_GRIDTOOLS_BRANCH $CLANG_GRIDTOOLS_REPOSITORY clang-gridtools

sarus pull $image

srun --job-name=dawn_PR \
    --time=00:45:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --ntasks-per-core=2 \
    --cpus-per-task=24 \
    --partition=cscsci \
    --constraint=gpu \
    --account=c14 \
    sarus run --mount=type=bind,source=$rootdir,destination=/usr/src/dawn \
        --mount=type=bind,source=$(pwd)/clang-gridtools,destination=/usr/src/clang-gridtools \
        $image \
        /usr/src/dawn/scripts/build-and-test /usr/src/dawn /usr/src/dawn-build /usr/local /usr/src/clang-gridtools \
            -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
            -DGridTools_DIR=/usr/local/lib/cmake \
            -DPROTOBUF_PYTHON_DIR=/usr/local/lib/python3.7/dist-packages \
            -GNinja
