#!/usr/bin/env bash
##===-------------------------------------------------------------------------------*- bash -*-===##
##                          _                      
##                         | |                     
##                       __| | __ ___      ___ ___  
##                      / _` |/ _` \ \ /\ / / '_  | 
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

# @brief Issue an error message to `stderr` and exit with 1
#
# @param $1   Message to print
function fatal_error() {
  2&> echo "error: $1" 
  exit 1
}

# @brief Build dawn and run the unittests
function build_and_test() {
  export CXX=${CXX_COMPILER}
  export CC=${C_COMPILER}

  $CC --version
  $CXX --version

  # Build dawn
  pushd "$(pwd)"

  mkdir build && cd build
  cmake .. -DCMAKE_CXX_COMPILER="$CXX"                                                             \
           -DCMAKE_C_COMPILER="$CC"                                                                \
        || fatal_error "failed to configure"
  make -j2 || fatal_error "failed to build"

  # Run unittests
  make test || fatal_error "failed to tests"

  popd

  # Everything ok!
  exit 0
}

build_and_test "$*"