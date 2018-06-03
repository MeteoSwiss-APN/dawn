#!/bin/bash

BASEPATH_SCRIPT=$(dirname "${0}")
source ${BASEPATH_SCRIPT}/machine_env.sh
source ${BASEPATH_SCRIPT}/env_${myhost}.sh

module load git
module load cmake
module load gcc/5.4.0-2.26
module load python/3.6.2-gmvolf-17.02

export BOOST_DIR=/scratch/cosuna/software/boost_1_59_0/

base_dir=$(pwd)
build_dir=${base_dir}/bundle/build

mkdir -p $build_dir
cd $build_dir

if [ ${myhost} == "kesch" ]; then
  PROTOBUFDIR="/scratch/jenkins/workspace/protobuf/slave/kesch/install/lib64/cmake/protobuf/"
elif [ ${myhost} == "daint" ]; then
  PROTOBUFDIR="/scratch/snx3000/jenkins/workspace/protobuf/slave/daint/install/lib64/cmake/protobuf/"
else
  echo" Error Machine not found: ${myhost}"
  exit 1
fi


cmake -DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_CXX_COMPILER=`which g++` -DCMAKE_C_COMPILER=`which gcc` -DBOOST_ROOT=${BOOST_DIR}  \
        -DProtobuf_DIR=${PROTOBUFDIR} ../
make -j2

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process  
