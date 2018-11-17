#!/bin/bash

module load git
module load /project/c14/software/modules/kesch/cmake/3.12.4
module load gcc/5.4.0-2.26
module load python/3.6.2-gmvolf-17.02
module load cudatoolkit/8.0.61

export CXX=`which g++`
export CC=`which gcc`
export BOOST_DIR=/project/c14/install/kesch/boost/boost_1_67_0/
export LLVM_DIR=/scratch/cosuna/software/clang/clang-3.8.1/install/

export SLURM_RESOURCES=('--gres=gpu:1')
export SLURM_PARTITION="debug"
export GPU_DEVICE="K80"

