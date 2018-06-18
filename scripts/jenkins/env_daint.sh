#!/bin/bash

module load CMake
module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/5.3.0
module load cudatoolkit

export BOOST_DIR=/project/c14/install/daint/boost/boost_1_67_0/
module load /project/c14/install/daint/clang/module_3.8.1
export CXX=`which g++`
export CC=`which gcc`

export SLURM_RESOURCES=('--gres=gpu:1 -C gpu --account=c14')
export SLURM_PARTITION="debug"
export GPU_DEVICE="P100"
