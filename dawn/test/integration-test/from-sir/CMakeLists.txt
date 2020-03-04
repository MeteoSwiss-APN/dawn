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

set(executable ${PROJECT_NAME}IntegrationTestOptimizerFromSIR)
add_executable(${executable}
  TestMain.cpp
  # TestComputeMaximumExtent.cpp
  # TestMultiStage.cpp
  # TestStage.cpp
)
target_link_libraries(${executable} PRIVATE DawnOptimizer DawnCompiler DawnUnittest gtest)
target_add_dawn_standard_props(${executable})
target_include_directories(${executable}
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
file(COPY input DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
gtest_discover_tests(${executable} TEST_PREFIX "Dawn::Integration::Optimizer::TestsFromSIR::"
  EXTRA_ARGS "${CMAKE_CURRENT_BINARY_DIR}/"
  DISCOVERY_TIMEOUT 30)
