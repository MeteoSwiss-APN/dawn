#!/bin/sh

script_path=$(dirname $(which $0))
root_dir=$script_path/../../

image=gtclang/dawn-env-cuda10.1-ubuntu18.04
build_jobs=24

# Fill myhost variable value
source $script_path/machine_env.sh
# Get machine-specific dependency locations
source $script_path/env_${myhost}.sh

# Clone clang-gridtools here because the image does not have the ssh key
if [ -z ${CLANG_GRIDTOOLS_REPOSITORY+x} ]; then
    CLANG_GRIDTOOLS_REPOSITORY=ssh://git@ssh.github.com:443/MeteoSwiss-APN/clang-gridtools.git
fi
if [ -z ${CLANG_GRIDTOOLS_BRANCH+x} ]; then
    CLANG_GRIDTOOLS_BRANCH=master
fi

git clone --depth 1 -b $CLANG_GRIDTOOLS_BRANCH $CLANG_GRIDTOOLS_REPOSITORY $root_dir/clang-gridtools

python -m venv dawn_venv

$root_dir/scripts/build-and-test \
    --dawn-build-dir $root_dir/build \
    --dawn-install-dir $root_dir/install \
    --clang-gridtools-source-dir $root_dir/clang-gridtools \
    --clang-gridtools-build-dir $root_dir/clang-gridtools-build \
    --parallel $build_jobs \
    --config $build_type \
    --python-venv dawn_venv \
    -DProtobuf_DIR=$PROTOBUFDIR \
    -DPROTOBUF_PYTHON_MODULE_DIR=$PROTOBUFDIR/../../../python \
    -DDAWN_REQUIRE_PYTHON=ON \
    -Datlas_DIR=${ATLAS_DIR}/lib/cmake/atlas \
    -Deckit_DIR=${ECKIT_DIR}/lib/cmake/eckit
