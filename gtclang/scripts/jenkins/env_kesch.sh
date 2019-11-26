#!/bin/bash

module load PE/17.06
module load git
module load /users/jenkins/easybuild/kesch/modules/all/cmake/3.12.4
module load python/3.6.2-gmvolf-17.02
module load cudatoolkit/8.0.61

export PROTOBUFDIR="/scratch/jenkins/workspace/protobuf/slave/kesch/install/lib64/cmake/protobuf/"
export CXX=`which g++`
export CC=`which gcc`
export BOOST_DIR=/project/c14/install/kesch/boost/boost_1_67_0/
export BOOST_ROOT=${BOOST_DIR}
export LLVM_DIR=/project/c14/install/kesch/clang/llvmorg-6.0.1

export SLURM_RESOURCES=('--gres=gpu:1')
export SLURM_PARTITION="debug"
export GPU_DEVICE="K80"

