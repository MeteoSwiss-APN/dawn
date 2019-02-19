#!/bin/bash

module rm CMake
module load /users/jenkins/easybuild/daint/haswell/modules/all/CMake/3.12.4

module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/5.3.0
module load cudatoolkit

export BOOST_DIR=/project/c14/install/daint/boost/boost_1_67_0/
module load /project/c14/install/daint/clang/module_6.0.0
export CXX=`which g++`
export CC=`which gcc`

export SLURM_RESOURCES=('--gres=gpu:1 -C gpu --account=c14')
export SLURM_PARTITION="debug"
export GPU_DEVICE="P100"
