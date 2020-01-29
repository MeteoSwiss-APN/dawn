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
  echo -e "-r clang gridtools repository to use"
  echo -e "-b clang gridtools branch to use"
  echo -e "-g gtclang install directory to use"
  echo -e "-p enabling performance checks"
  echo -e "-h shows this help"
  exit 1
}
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
while getopts b:r:g:ph flag; do
  case $flag in
    r)
      CLANG_GRIDTOOLS_REPOSITORY=$OPTARG
      ;;
    b)
      CLANG_GRIDTOOLS_BRANCH=$OPTARG
      ;;
    g)
      GTCLANG_INSTALL_DIR=$OPTARG
      ;;
    p)
      RUN_PERFTETS=true
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

if [ -z ${CLANG_GRIDTOOLS_REPOSITORY+x} ]; then
  # ssh port is blocked on compute nodes, go via https port
  export CLANG_GRIDTOOLS_REPOSITORY=ssh://git@ssh.github.com:443/MeteoSwiss-APN/clang-gridtools.git
fi
if [ -z ${CLANG_GRIDTOOLS_BRANCH+x} ]; then
  export CLANG_GRIDTOOLS_BRANCH=master
fi

cd ${root_dir}
git clone ${CLANG_GRIDTOOLS_REPOSITORY} --depth=1 --branch=${CLANG_GRIDTOOLS_BRANCH}

source_dir=${root_dir}/clang-gridtools
build_dir=${source_dir}/build

mkdir -p $build_dir
cd $build_dir

if [ -z ${GTCLANG_INSTALL_DIR+x} ]; then
  export GTCLANG_INSTALL_DIR=${root_dir}/install #try this
fi

if [ -z ${PROTOBUFDIR+x} ]; then
 echo "PROTOBUFDIR needs to be set in the machine env"
fi


CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${build_type} \
            -DCMAKE_PREFIX_PATH=${GTCLANG_INSTALL_DIR} \
            -DProtobuf_DIR=${PROTOBUFDIR} \
            -DPROTOBUF_PYTHON_MODULE_DIR=${PROTOBUFDIR}/../../../python \
            -DGridTools_DIR=${GTCLANG_INSTALL_DIR}/lib/cmake \
            "

cmake ${CMAKE_ARGS} ${source_dir}

if [ -z ${PARALLEL_BUILD_JOBS+x} ]; then
  PARALLEL_BUILD_JOBS=8
fi

echo "Building with ${PARALLEL_BUILD_JOBS} jobs."
cmake --build . --parallel ${PARALLEL_BUILD_JOBS}

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process

if [ -z ${RUN_PERFTETS+x} ]; then
  echo "do not run performance tests"
else
  if [ "${target}" = "cuda" ]; then
    export ENABLE_CUDA_GPU=true
  else
    export ENABLE_GT_GPU=true
  fi
 cd $source_dir
 bash scripts/jenkins/run_perftests.sh -b $build_dir 
fi
