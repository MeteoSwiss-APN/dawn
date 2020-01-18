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

set(executable ${PROJECT_NAME}UnittestOptimizer)
add_executable(${executable}
  TestColoringAlgorithm.cpp
  TestEnvironment.h
  TestGraph.cpp
  TestIsDAGAlgorithm.cpp
  TestPartitionAlgorithm.cpp
  TestMain.cpp
  TestSCCAlgorithm.cpp
)
target_link_libraries(${executable} PRIVATE DawnOptimizer DawnUnittest gtest)
target_add_dawn_standard_props(${executable})
gtest_discover_tests(${executable} TEST_PREFIX "${executable}::" DISCOVERY_TIMEOUT 30)

add_subdirectory(TestsFromSIR)
add_subdirectory(Passes)
