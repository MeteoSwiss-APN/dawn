#!/bin/bash

set -e

module load git
module load cmake
module load gcc/5.4.0-2.26
module load python/3.6.2-gmvolf-17.02

export BOOST_DIR=/scratch/cosuna/software/boost_1_59_0/

base_dir=$(pwd)
build_dir=${base_dir}/bundle/build

mkdir -p $build_dir
cd $build_dir


cmake -DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_CXX_COMPILER=`which g++` -DCMAKE_C_COMPILER=`which gcc` -DBOOST_ROOT=${BOOST_DIR} -DGTCLANG_ENABLE_GRIDTOOLS=ON \
        -DProtobuf_DIR=/scratch/cosuna/software/protobuf/3.4.0/lib/cmake/protobuf/  -DLLVM_ROOT=/scratch/cosuna/software/clang/clang-3.8.1/install/ ../
make -j2

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process  
