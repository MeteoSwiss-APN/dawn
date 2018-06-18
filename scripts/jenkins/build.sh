#!/bin/bash

BASEPATH_SCRIPT=$(dirname "${0}")
source ${BASEPATH_SCRIPT}/machine_env.sh
source ${BASEPATH_SCRIPT}/env_${myhost}.sh

base_dir=$(pwd)
build_dir=${base_dir}/bundle/build

mkdir -p $build_dir
cd $build_dir

if [ -z ${BOOST_DIR+x} ]; then 
 echo "BOOST_DIR needs to be set in the machine env"
fi

if [ -z ${PROTOBUFDIR+x} ]; then 
 echo "PROTOBUFDIF needs to be set in the machine env"
fi

cmake -DCMAKE_BUILD_TYPE=${build_type} -DBOOST_ROOT=${BOOST_DIR}  \
        -DProtobuf_DIR=${PROTOBUFDIR} ../
make -j2

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process  
