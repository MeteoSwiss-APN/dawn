#!/bin/bash

module rm CMake
module load /users/jenkins/easybuild/daint/haswell/modules/all/CMake/3.12.4

module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/7.3.0
module load cudatoolkit/9.2.148_3.19-6.0.7.1_2.1__g3d9acc8

export BOOST_DIR=/project/c14/install/daint/boost/boost_1_67_0/
module load /project/c14/install/daint/clang/module_6.0.0-gcc-7.3.0
export CXX=`which g++`
export CC=`which gcc`

export SLURM_RESOURCES=('--gres=gpu:1 -C gpu --account=g110')
export SLURM_PARTITION="debug"
export GPU_DEVICE="P100"
