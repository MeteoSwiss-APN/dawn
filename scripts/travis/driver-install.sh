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

this_script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$this_script_dir/install.sh"
install_driver $*

export CXX=${CXX_COMPILER}
export CC=${C_COMPILER}

$CC --version
$CXX --version

# Build dawn
pushd "$(pwd)"

export PYTHON_DIR=/opt/python/3.5.3

cd bundle
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER="$CXX"                                                               \
         -DCMAKE_C_COMPILER="$CC"                                                                  \
         -DCMAKE_BUILD_TYPE="$CONFIG"                                                              \
         -DPYTHON_EXECUTABLE="$PYTHON_DIR/bin/python3"                                             \
      || fatal_error "failed to configure"
make -j2 protobuf || fatal_error "failed to build"

