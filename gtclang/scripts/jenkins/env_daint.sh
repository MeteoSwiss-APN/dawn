#!/bin/bash

module load CMake

module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/7.3.0
module load cudatoolkit/10.1.105_3.27-7.0.1.1_4.1__ga311ce7

export BOOST_DIR=/project/c14/install/daint/boost/boost_1_67_0/
module load /project/c14/install/daint/clang/module_6.0.0-gcc-7.3.0
export CXX=`which g++`
export CC=`which gcc`

export SLURM_RESOURCES=('--gres=gpu:1 -C gpu --account=g110')
export SLURM_PARTITION="debug"
export GPU_DEVICE="P100"
