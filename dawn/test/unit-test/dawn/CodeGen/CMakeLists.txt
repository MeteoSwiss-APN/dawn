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
include(GoogleTest)

file(COPY input DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY reference DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(executable ${PROJECT_NAME}UnittestCodeGen)
add_executable(${executable} TestCodeGen.cpp)
target_add_dawn_standard_props(${executable})
target_link_libraries(${executable} DawnUnittest gtest gtest_main)
gtest_discover_tests(${executable} TEST_PREFIX "Dawn::Unit::CodeGen::"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DISCOVERY_TIMEOUT 30
)
