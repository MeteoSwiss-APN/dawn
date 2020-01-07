#!/bin/bash

set -e

root_dir=$(pwd)
BASEPATH_SCRIPT=$(dirname $(realpath -s $0))
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
  echo -e "-h Shows this help"
  exit 1
}
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
while getopts b: flag; do
  case $flag in
    b)
      CLANG_GRIDTOOLS_BRANCH=$OPTARG
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

if [ -z ${CLANG_GRIDTOOLS_BRANCH+x} ]; then
  export CLANG_GRIDTOOLS_BRANCH=master
fi

cd ${root_dir}
git clone git@github.com:MeteoSwiss-APN/clang-gridtools.git --depth=1 --branch=${CLANG_GRIDTOOLS_BRANCH}

source_dir=${root_dir}/clang-gridtools
build_dir=${source_dir}/build

mkdir -p $build_dir
cd $build_dir

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${build_type} \
            -DProtobuf_DIR=${PROTOBUFDIR} \
            -DPROTOBUF_PYTHON_MODULE_DIR=${PROTOBUFDIR}/../../../python \
            -DGTCLANG_REQUIRE_UNSTRUCTURED_TESTING=ON \
            -DDAWN_REQUIRE_PYTHON_TESTING=ON \
            -Datlas_DIR=${ATLAS_DIR}/lib/cmake/atlas \
            -Deckit_DIR=${ECKIT_DIR}/lib/cmake/eckit \
            "

cmake ${CMAKE_ARGS} ${source_dir}

if [ -z ${PARALLEL_BUILD_JOBS+x} ]; then
  PARALLEL_BUILD_JOBS=8
fi

echo "Building with ${PARALLEL_BUILD_JOBS} jobs."
cmake --build . --parallel ${PARALLEL_BUILD_JOBS}

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process
