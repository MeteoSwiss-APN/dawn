#!/bin/bash

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export TRAVIS_BUILD_DIR=${SCRIPT_DIR}/../../bundle/build
export CACHE_DIR=${TRAVIS_BUILD_DIR}/deps/install
export CMAKE_VERSION=3.8.1

export CXX_COMPILER=g++-5
export C_COMPILER=gcc-5

bash ${SCRIPT_DIR}/driver-install.sh -i ${CACHE_DIR} -b cmake

bash ${SCRIPT_DIR}/driver-test.sh

