#!/bin/bash

set -e

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

source_dir=${BASEPATH_SCRIPT}/../..
base_dir=$(pwd)
build_dir=${base_dir}/build

# python tests
mkdir -p $build_dir
cd $build_dir
python -m venv dawn_venv
source dawn_venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -e ${base_dir}/dawn/ -v

cd ${base_dir}/dawn/examples/python
bash run.sh
echo "PYTHON RUN TESTS SUCCESFUL!"

python -m pytest -v ${base_dir}/dawn/test/unit-test/test_dawn4py/
echo "PYTHON PYTEST SUCCESFUL!"

base_dir=$(pwd)
rm -rf $build_dir

mkdir -p $build_dir
cd $build_dir

if [ -z ${PROTOBUFDIR+x} ]; then
 echo "PROTOBUFDIR needs to be set in the machine env"
fi

#TODO -DDAWN_BUNDLE_JAVA=ON
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${build_type} \
            -DProtobuf_DIR=${PROTOBUFDIR} \
            -DPROTOBUF_PYTHON_MODULE_DIR=${PROTOBUFDIR}/../../../python \
            -DGTCLANG_REQUIRE_UNSTRUCTURED_TESTING=ON \
            -DDAWN_REQUIRE_PYTHON_TESTING=ON \
            -Datlas_DIR=${ATLAS_DIR}/lib/cmake/atlas \
            -Deckit_DIR=${ECKIT_DIR}/lib/cmake/eckit \
            "

if [ -z ${INSTALL_DIR+x} ]; then
  INSTALL_DIR=${base_dir}/install
fi

CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"

cmake ${CMAKE_ARGS} ${source_dir}

if [ -z ${PARALLEL_BUILD_JOBS+x} ]; then
  PARALLEL_BUILD_JOBS=8
fi

echo "Building with ${PARALLEL_BUILD_JOBS} jobs."
cmake --build . --parallel ${PARALLEL_BUILD_JOBS}

# Run unittests
ctest -C ${build_type} --output-on-failure --force-new-ctest-process

cmake --build . --parallel ${PARALLEL_BUILD_JOBS} --target install
