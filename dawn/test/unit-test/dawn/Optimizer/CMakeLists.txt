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
target_link_libraries(DawnUnittestOptimizer Dawn gtest)
target_compile_options(DawnUnittestOptimizer PRIVATE ${DAWN_CXX_FLAGS})
target_include_directories(DawnUnittestOptimizer PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(DawnUnittestOptimizer PROPERTIES ${DAWN_TARGET_PROPERTIES})
add_test(NAME DawnUnittestOptimizer
  COMMAND DawnUnittestOptimizer
)

add_subdirectory(TestsFromSIR)
add_subdirectory(Passes)
