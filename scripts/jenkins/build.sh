#!/bin/bash

module load git/2.8.4
module load CMake/3.8.1
module load GCC/5.3.0-binutils-2.25
#Avoid conflicts complains
module load Python/3.5.0-gmvolf-15.11 >& /dev/null

export BOOST_DIR=/scratch/cosuna/software/boost_1_59_0/

base_dir=$(pwd)
build_dir=${base_dir}/bundle/build

mkdir -p $build_dir
cd $build_dir

export BOOST_DIR=

cmake -DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_CXX_COMPILER=`which g++` -DCMAKE_C_COMPILER=`which gcc` -DBOOST_ROOT=$(BOOST_DIR) -DGTCLANG_ENABLE_GRIDTOOLS=ON \
        -DProtobuf_DIR=/scratch/cosuna/software/protobuf/3.4.0/lib/cmake/protobuf/  -DLLVM_ROOT=/scratch/cosuna/software/clang/clang-3.8.1/install/ ../
make -j2

# Run unittests
ctest -C ${build_type} --output-on-failure --force-new-ctest-process  
