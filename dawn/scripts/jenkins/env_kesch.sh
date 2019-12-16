#!/bin/bash


module load PE/17.06
module load git
module load /users/jenkins/easybuild/kesch/modules/all/cmake/3.12.4
module load python/3.6.2-gmvolf-17.02

export CXX=`which g++`
export CC=`which gcc`
export BOOST_DIR=/project/c14/install/kesch/boost/boost_1_67_0/
export PROTOBUFDIR="/project/c14/install/kesch/protobuf/lib64/cmake/protobuf/"

