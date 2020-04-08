#!/bin/sh

script_path=$(dirname $(which $0))
root_dir=$script_path/../../

image=gtclang/dawn-env-cuda10.1-ubuntu18.04
build_jobs=24

module load daint-gpu
module load sarus

# Clone clang-gridtools here because the image does not have the ssh key
if [ -z ${CLANG_GRIDTOOLS_REPOSITORY+x} ]; then
    CLANG_GRIDTOOLS_REPOSITORY=git@github.com:MeteoSwiss-APN/clang-gridtools.git
fi
if [ -z ${CLANG_GRIDTOOLS_BRANCH+x} ]; then
    CLANG_GRIDTOOLS_BRANCH=master
fi

git clone --depth 1 -b $CLANG_GRIDTOOLS_BRANCH $CLANG_GRIDTOOLS_REPOSITORY $(pwd)/clang-gridtools

docker pull $image

srun --job-name=dawn_PR \
    --time=00:45:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --ntasks-per-core=2 \
    --cpus-per-task=$build_jobs \
    --partition=cscsci \
    --constraint=gpu \
    --account=c14 \
    docker run -v $root_dir:/usr/src/dawn \
    $(pwd)/clang-gridtools:/usr/src/clang-gridtools \
    $image \
    /usr/src/dawn/scripts/build-and-test \
    --dawn-build-dir /usr/src/dawn-build \
    --dawn-install-dir /usr/local/dawn \
    --clang-gridtools-source-dir /usr/src/clang-gridtools \
    --clang-gridtools-build-dir /usr/src/clang-gridtools-build \
    --parallel $build_jobs \
    --config $build_type \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
    -DProtobuf_DIR=/usr/local/protobuf/lib/cmake/protobuf \
    -DPROTOBUF_PYTHON_DIR=/usr/local/lib/python3.7/dist-packages \
    -DGridTools_DIR=/usr/local/gridtools/lib/cmake \
    -Datlas_DIR=/usr/local/atlas/lib/cmake/atlas \
    -Deckit_DIR=/usr/local/eckit/lib/cmake/eckit \
    -GNinja

# sarus pull $image

# srun --job-name=dawn_PR \
#     --time=00:45:00 \
#     --nodes=1 \
#     --ntasks-per-node=1 \
#     --ntasks-per-core=2 \
#     --cpus-per-task=$build_jobs \
#     --partition=cscsci \
#     --constraint=gpu \
#     --account=c14 \
#     sarus run --mount=type=bind,source=$root_dir,destination=/usr/src/dawn \
#     --mount=type=bind,source=$(pwd)/clang-gridtools,destination=/usr/src/clang-gridtools \
#     $image \
#     /usr/src/dawn/scripts/build-and-test \
#     --dawn-build-dir /usr/src/dawn-build \
#     --dawn-install-dir /usr/local/dawn \
#     --clang-gridtools-source-dir /usr/src/clang-gridtools \
#     --clang-gridtools-build-dir /usr/src/clang-gridtools-build \
#     --parallel $build_jobs \
#     --config $build_type \
#     -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
#     -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
#     -DProtobuf_DIR=/usr/local/protobuf/lib/cmake/protobuf \
#     -DPROTOBUF_PYTHON_DIR=/usr/local/lib/python3.7/dist-packages \
#     -DGridTools_DIR=/usr/local/gridtools/lib/cmake \
#     -Datlas_DIR=/usr/local/atlas/lib/cmake/atlas \
#     -Deckit_DIR=/usr/local/eckit/lib/cmake/eckit \
#     -GNinja
