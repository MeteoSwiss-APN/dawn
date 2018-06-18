#!/bin/bash

module load CMake
module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/5.3.0

export BOOST_DIR=/project/c14/install/daint/boost/boost_1_67_0/
export PROTOBUFDIR="/scratch/snx3000/jenkins/workspace/protobuf/slave/daint/install/lib64/cmake/protobuf/"
export CXX=`which g++`
export CC=`which gcc`
