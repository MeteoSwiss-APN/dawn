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

add_executable(DawnUnittestOptimizer
  TestColoringAlgorithm.cpp
  TestEnvironment.h
  TestGraph.cpp
  TestIsDAGAlgorithm.cpp
  TestPartitionAlgorithm.cpp
  TestMain.cpp
  TestSCCAlgorithm.cpp
)
target_link_libraries(DawnUnittestOptimizer PRIVATE DawnOptimizer DawnUnittest gtest)
target_add_dawn_standard_props(DawnUnittestOptimizer)
add_test(NAME DawnUnittestOptimizer
  COMMAND DawnUnittestOptimizer
)

add_subdirectory(TestsFromSIR)
add_subdirectory(Passes)
