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

set(executable ${PROJECT_NAME}UnittestOptimizerFromSIR)
add_executable(${executable}
  TestMain.cpp
  TestComputeMaximumExtent.cpp
  TestMultiStage.cpp
  TestStage.cpp
)
target_link_libraries(${executable} PRIVATE DawnOptimizer DawnCompiler DawnUnittest gtest)
target_add_dawn_standard_props(${executable})
target_include_directories(${executable}
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
add_test(NAME ${executable}
  COMMAND ${executable} "${CMAKE_CURRENT_LIST_DIR}/../Passes"
)
