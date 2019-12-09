#!/bin/bash

module rm CMake
module load /users/jenkins/easybuild/daint/haswell/modules/all/CMake/3.12.4
module load cray-python/3.6.5.7

module swap PrgEnv-cray PrgEnv-gnu

export BOOST_DIR=/project/c14/install/daint/boost/boost_1_67_0/
export PROTOBUFDIR="/project/c14/install/daint/protobuf/lib64/cmake/protobuf/"
export CXX=`which g++`
export CC=`which gcc`
