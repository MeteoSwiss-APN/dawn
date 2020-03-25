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

file(COPY input DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY reference DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(Cuda)
add_subdirectory(Naive)
add_subdirectory(Naive-ico)
add_subdirectory(samples)
add_subdirectory(reference)