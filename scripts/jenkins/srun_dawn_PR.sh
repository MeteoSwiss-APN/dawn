#!/bin/sh

module load daint-gpu
module load sarus

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
    sarus run --mount=type=bind,source=$rootdir,destination=/usr/src/dawn $image \
        /usr/src/dawn/scripts/build-and-test /usr/src/dawn /usr/src/dawn-build /usr/local \
            -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
            -DGridTools_DIR=/usr/local/lib/cmake \
            -DPROTOBUF_PYTHON_DIR=/usr/local/lib/python3.7/dist-packages \
            -GNinja
