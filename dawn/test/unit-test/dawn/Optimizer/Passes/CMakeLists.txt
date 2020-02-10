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

set(executable ${PROJECT_NAME}UnittestOptimizerPasses)
add_executable(${executable}
  TestMain.cpp
  TestPassComputeStageExtents.cpp
  TestComputeMaxExtent.cpp
  TestPassCaching.cpp
  TestPassLocalVarType.cpp
  TestPassSetBoundaryCondition.cpp
  TestFieldAccessIntervals.cpp
  TestTemporaryToFunction.cpp
  TestIntervalPartitioner.cpp
)
target_link_libraries(${executable} PRIVATE DawnOptimizer DawnCompiler DawnUnittest gtest)
target_add_dawn_standard_props(${executable})
target_include_directories(${executable}
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
gtest_discover_tests(${executable} TEST_PREFIX "Dawn::Unit::Optimizer::Passes::"
  EXTRA_ARGS ${CMAKE_CURRENT_LIST_DIR}
  DISCOVERY_TIMEOUT 30)
