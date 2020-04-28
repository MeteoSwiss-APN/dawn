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

set(executable ${PROJECT_NAME}UnittestValidator)
add_executable(${executable}
  TestUnstructuredDimensionChecker.cpp
  TestGridTypeChecker.cpp
  TestIntegrityChecker.cpp
  TestWeightChecker.cpp
)
target_link_libraries(${executable} DawnValidator DawnUnittest gtest gtest_main)
target_add_dawn_standard_props(${executable})
gtest_discover_tests(${executable} TEST_PREFIX "Dawn::Unit::Validator::" DISCOVERY_TIMEOUT 30)
file(COPY input DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
