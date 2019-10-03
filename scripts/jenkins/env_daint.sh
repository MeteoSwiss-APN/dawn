#!/bin/bash

module rm CMake
module load /users/jenkins/easybuild/daint/haswell/modules/all/CMake/3.12.4
module load cray-python/3.6.5.3

module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/7.3.0

export BOOST_DIR=/project/c14/install/daint/boost/boost_1_67_0/
export PROTOBUFDIR="/scratch/snx3000/jenkins/workspace/protobuf/slave/daint/install/lib64/cmake/protobuf/"
export CXX=`which g++`
export CC=`which gcc`
