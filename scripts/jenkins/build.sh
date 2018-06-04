#!/bin/bash

BASEPATH_SCRIPT=$(dirname "${0}")
source ${BASEPATH_SCRIPT}/machine_env.sh
source ${BASEPATH_SCRIPT}/env_${myhost}.sh

base_dir=$(pwd)
build_dir=${base_dir}/bundle/build

mkdir -p $build_dir
cd $build_dir

if [ ${myhost} == "kesch" ]; then
  BOOST_DIR="/scratch/jenkins/workspace/boost_build/boost_version/1.67.0/slave/kesch/boost_1_67_0/"
  PROTOBUFDIR="/scratch/jenkins/workspace/protobuf/slave/kesch/install/lib64/cmake/protobuf/"
elif [ ${myhost} == "daint" ]; then
  BOOST_DIR="/scratch/snx3000/jenkins/workspace/boost_build/boost_version/1.67.0/slave/daint/boost_1_67_0/"
  PROTOBUFDIR="/scratch/snx3000/jenkins/workspace/protobuf/slave/daint/install/lib64/cmake/protobuf/"
else
  echo" Error Machine not found: ${myhost}"
  exit 1
fi


cmake -DCMAKE_BUILD_TYPE=${build_type} -DBOOST_ROOT=${BOOST_DIR}  \
        -DProtobuf_DIR=${PROTOBUFDIR} ../
make -j2

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process  
