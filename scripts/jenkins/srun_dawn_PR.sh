#!/bin/sh

set -e

scriptpath=$(dirname $(realpath -s $0))
rootdir=$scriptpath/../../
image=gtclang/dawn-env-ubuntu19.10

module load daint-gpu
module load sarus

sarus images
sarus pull $image
#sarus rmi gtclang/dawn-env-ubuntu19.04
sarus images

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
     cmake -S /usr/src/dawn -B /usr/src/dawn/build \
    -DBUILD_TESTING=ON \
    -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DGridTools_DIR=/usr/local/lib/cmake \
    -DPROTOBUF_PYTHON_DIR=/usr/local/lib/python3.7/dist-packages \
    -GNinja \
    && cmake --build /usr/src/dawn/build --parallel 24 --target install \
    && python -m pip install -e /usr/src/dawn/dawn \
    && cd /usr/src/dawn/build && ctest --progress --output-on-failure

#     $BASEPATH_SCRIPT/dawn_PR.sh "$@"
