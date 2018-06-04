#!/bin/bash

module load CMake
module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/5.3.0

export BOOST_DIR=/project/c14/install/daint/boost/boost_1_67_0/
module load /users/vogtha/modules/compilers/clang/3.8.1
export CXX=${LLVM_DIR}/bin/clang++
export CC=${LLVM_DIR}/bin/clang

