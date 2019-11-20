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


add_executable(DawnUnittestOptimizerFromSIR
  TestMain.cpp
  TestComputeMaximumExtent.cpp
  TestMultiStage.cpp
  TestStage.cpp
)
target_link_libraries(DawnUnittestOptimizerFromSIR PRIVATE DawnOptimizer DawnCompiler DawnUnittest gtest)
target_add_dawn_standard_props(DawnUnittestOptimizerFromSIR)
target_include_directories(DawnUnittestOptimizerFromSIR
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
add_test(NAME DawnUnittestOptimizerFromSIR
  COMMAND DawnUnittestOptimizerFromSIR "${CMAKE_CURRENT_LIST_DIR}/../Passes"
)
