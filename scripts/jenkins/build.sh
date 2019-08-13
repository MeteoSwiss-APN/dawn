#!/bin/bash

# set -e

BASEPATH_SCRIPT=$(dirname "${0}")
source ${BASEPATH_SCRIPT}/machine_env.sh
source ${BASEPATH_SCRIPT}/env_${myhost}.sh

if [ -z ${myhost+x} ]; then
  echo "myhost is unset"
  exit 1
fi

SCRIPT=`basename $0`

function help {
  echo -e "Basic usage:$SCRIPT "\\n
  echo -e "The following switches are recognized. $OFF "
  echo -e "-i sets the installation directory"
  echo -e "-h Shows this help"
  exit 1
}
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
while getopts i: flag; do
  case $flag in
    i)
      INSTALL_DIR=$OPTARG
      ;;
    h)
      help
      ;;
    \?) #unrecognized option - show help
      echo -e \\n"Option -${BOLD}$OPTARG${OFF} not allowed."
      help
      ;;
  esac
done

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
CMAKE_ARGS="-DDAWN_BUNDLE_PYTHON=ON -DDAWN_BUNDLE_JAVA=ON -DDAWN_PYTHON_EXAMPLES=ON -DCMAKE_BUILD_TYPE=${build_type} -DBOOST_ROOT=${BOOST_DIR}  \
        -DProtobuf_DIR=${PROTOBUFDIR}"

if [ -n ${INSTALL_DIR} ]; then
  rm -rf ${INSTALL_DIR}
  CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
fi

cmake ${CMAKE_ARGS} ../
make -j2 install

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process  
