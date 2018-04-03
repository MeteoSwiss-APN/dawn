#!/bin/bash

set -e

module load git
module load cmake
module load gcc/5.4.0-2.26
module load python/3.6.2-gmvolf-17.02

export BOOST_DIR=/scratch/cosuna/software/boost_1_59_0/


SCRIPT=`basename $0`

function help {
  echo -e "Basic usage:$SCRIPT "\\n
  echo -e "The following switches are recognized. $OFF "
  echo -e "-i sets the installation directory"
  echo -e "-h Shows this help"
  exit 1
}

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

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_CXX_COMPILER=`which g++` -DCMAKE_C_COMPILER=`which gcc` -DBOOST_ROOT=${BOOST_DIR} -DGTCLANG_ENABLE_GRIDTOOLS=ON \
        -DProtobuf_DIR=/scratch/cosuna/software/protobuf/3.4.0/lib/cmake/protobuf/  -DLLVM_ROOT=/scratch/cosuna/software/clang/clang-3.8.1/install/ ../"

if [ -z ${INSTALL_DIR} ]; then
  cmake ${CMAKE_ARGS}
else
  cmake ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
fi

make -j2 install

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process  
