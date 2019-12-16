#!/bin/bash

set -e

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
# Setup for the python tests:
mkdir -p $build_dir
cd $build_dir
export TMPDIR=${build_dir}/temp
mkdir -p $TMPDIR
python -m venv dawn_venv
source dawn_venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -e ${base_dir} -v

cd ${base_dir}/examples/python
bash run.sh
pytest -v ${base_dir}/test/unit-test/test_dawn4py/

# clean up the build directory for the c++ tests
rm -rf $build_dir

# Test for the c++ side:
mkdir -p $build_dir
cd $build_dir

if [ -z ${PROTOBUFDIR+x} ]; then
 echo "PROTOBUFDIF needs to be set in the machine env"
fi
CMAKE_ARGS="-DDAWN_BUNDLE_PYTHON=ON -DDAWN_BUNDLE_JAVA=ON -DDAWN_PYTHON_EXAMPLES=ON -DCMAKE_BUILD_TYPE=${build_type}  \
        -DProtobuf_DIR=${PROTOBUFDIR} -DPROTOBUF_PYTHON_INSTALL=${PROTOBUFDIR}/../../../python"

if [ -n ${INSTALL_DIR} ]; then
  rm -rf ${INSTALL_DIR}
  CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
fi

cmake ${CMAKE_ARGS} ../
make -j8 install

# Run unittests
ctest -VV -C ${build_type} --output-on-failure --force-new-ctest-process
