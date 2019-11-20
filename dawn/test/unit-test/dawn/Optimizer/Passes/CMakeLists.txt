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


add_executable(DawnUnittestOptimizerPasses
  TestMain.cpp
  TestPassComputeStageExtents.cpp
  TestComputeMaxExtent.cpp
  TestPassSetBoundaryCondition.cpp
  TestFieldAccessIntervals.cpp
  TestTemporaryToFunction.cpp
)
target_link_libraries(DawnUnittestOptimizerPasses PRIVATE DawnOptimizer DawnCompiler DawnUnittest gtest)
target_add_dawn_standard_props(DawnUnittestOptimizerPasses)
target_include_directories(DawnUnittestOptimizerPasses
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
add_test(NAME DawnUnittestOptimizerPasses
  COMMAND DawnUnittestOptimizerPasses "${CMAKE_CURRENT_LIST_DIR}"
)
