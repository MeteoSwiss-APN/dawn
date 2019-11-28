#!/bin/bash

module rm CMake
module load /users/jenkins/easybuild/daint/haswell/modules/all/CMake/3.12.4

module swap PrgEnv-cray PrgEnv-gnu
module load cray-python/3.6.5.7
module load cudatoolkit/10.1.105_3.27-7.0.1.1_4.1__ga311ce7
module load /project/c14/install/daint/clang/module_6.0.1-gcc-8.3.0

export CXX=`which g++`
export CC=`which gcc`

export BOOST_DIR=/project/c14/install/daint/boost/boost_1_67_0/
export PROTOBUFDIR="/scratch/snx3000/jenkins/workspace/protobuf/slave/daint/install/lib64/cmake/protobuf/"
export ATLAS_DIR=/project/c14/install/daint/atlas_install
export ECKIT_DIR=/project/c14/install/daint/eckit_install

export SLURM_RESOURCES=('--gres=gpu:1 -C gpu --account=g110')
export SLURM_PARTITION="debug"
export CUDA_ARCH="sm_60"
