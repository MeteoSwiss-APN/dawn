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
  TestPassCaching.cpp
  TestPassLocalVarType.cpp
  TestPassIntervalPartitioning.cpp
  TestPassFieldVersioning.cpp
  TestPassMultiStageSplitter.cpp
  TestPassRemoveScalars.cpp
  TestPassSetCaches.cpp
  TestPassSetNonTempCaches.cpp
  TestPassSetStageLocationType.cpp
  TestPassSimplifyStatements.cpp
  TestPassStageMerger.cpp
  TestPassStageSplitAllStatements.cpp
  TestPassStageReordering.cpp
  TestPassTemporaryMerger.cpp
  TestPassTemporaryType.cpp
  TestTemporaryToFunction.cpp
)
target_link_libraries(${executable} PRIVATE DawnOptimizer DawnCompiler DawnAST DawnUnittest gtest gtest_main)
target_add_dawn_standard_props(${executable})
target_include_directories(${executable}
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
file(COPY input DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
gtest_discover_tests(${executable} TEST_PREFIX "Dawn::Unit::Optimizer::"
  DISCOVERY_TIMEOUT 30
)

  add_subdirectory(samples)
