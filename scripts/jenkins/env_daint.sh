#!/bin/bash

module load CMake
module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/5.3.0

LLVM_DIR=/scratch//snx3000/jenkins/workspace/clang/slave/daint/clang+llvm-3.8.1-x86_64-linux-gnu-ubuntu-16.04/
export CXX=${LLVM_DIR}/bin/clang++
export CC=${LLVM_DIR}/bin/clang

