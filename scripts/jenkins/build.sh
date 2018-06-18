#!/bin/bash

set -e

BASEPATH_SCRIPT=$(dirname "${0}")
source ${BASEPATH_SCRIPT}/machine_env.sh
source ${BASEPATH_SCRIPT}/env_${myhost}.sh


if [ -z ${SLURM_RESOURCES+x} ]; then 
  echo "SLURM_RESOURCES is unset"
fi
if [ -z ${myhost+x} ]; then
  echo "myhost is unset"
fi

SCRIPT=`basename $0`

function help {
  echo -e "Basic usage:$SCRIPT "\\n
  echo -e "The following switches are recognized. $OFF "
  echo -e "-i sets the installation directory"
  echo -e "-g gpu build" 
  echo -e "-h Shows this help"
  echo -e "-d <path> path to dawn"
  exit 1
}

ENABLE_GPU=false

while getopts i:gd: flag; do
  case $flag in
    i)
      INSTALL_DIR=$OPTARG
      ;;
    h)
      help
      ;;
    g)
      ENABLE_GPU=true
      ;;
    d) 
      DAWN_PATH=$OPTARG
      ;;
    \?) #unrecognized option - show help
      echo -e \\n"Option -${BOLD}$OPTARG${OFF} not allowed."
      help
      ;;
  esac
done

if [ ${myhost} == "kesch" ]; then
  PROTOBUFDIR="/scratch/jenkins/workspace/protobuf/slave/kesch/install/lib64/cmake/protobuf/"
elif [ ${myhost} == "daint" ]; then
  PROTOBUFDIR="/scratch/snx3000/jenkins/workspace/protobuf/slave/daint/install/lib64/cmake/protobuf/"
else
  echo" Error Machine not found: ${myhost}"
  exit 1
fi

base_dir=$(pwd)
build_dir=${base_dir}/bundle/build

mkdir -p $build_dir
cd $build_dir

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${build_type} -DBOOST_ROOT=${BOOST_DIR} -DGTCLANG_ENABLE_GRIDTOOLS=ON \
        -DProtobuf_DIR=${PROTOBUFDIR}  -DLLVM_ROOT=${LLVM_DIR}" 

if [ -n ${INSTALL_DIR} ]; then
  CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
fi
if [ ! -z ${DAWN_PATH} ]; then
  CMAKE_ARGS="${CMAKE_ARGS} -Ddawn_DIR=${DAWN_PATH}"
fi

if [ "$ENABLE_GPU" = true ]; then
  cmake ${CMAKE_ARGS} -DGTCLANG_BUILD_EXAMPLES_WITH_GPU=ON -DCTEST_CUDA_SUBMIT=ON -DGTCLANG_SLURM_RESOURCES="${SLURM_RESOURCES[@]}" -DGTCLANG_SLURM_PARTITION=${SLURM_PARTITION} -DGPU_DEVICE=${GPU_DEVICE} ../
else
  cmake ${CMAKE_ARGS} ../
fi

nice make -j6 install VERBOSE=1

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process  
