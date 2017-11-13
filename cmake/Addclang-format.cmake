##===------------------------------------------------------------------------------*- CMake -*-===##
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

find_package(clang-format EXACT 3.8)

dawn_export_package(
  NAME clang-format
  FOUND ${CLANG-FORMAT_FOUND}
  VERSION ${CLANG-FORMAT_VERSION}
  EXECUTABLE ${CLANG-FORMAT_EXECUTABLE}
)
