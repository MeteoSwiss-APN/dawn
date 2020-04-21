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

set(executable ${PROJECT_NAME}UnittestIIR)
add_executable(${executable}
  TestComputeStageExtents.cpp
  TestDependencyGraphAccesses.cpp
  TestExtent.cpp
  TestField.cpp
  TestFieldAccessIntervals.cpp
  TestInterval.cpp
  TestIntervalAlgorithms.cpp
  TestIIRNode.cpp
  TestIIRNodeIterator.cpp
  TestMultiInterval.cpp
  TestStencil.cpp
  TestIIRSerializer.cpp
)
target_link_libraries(${executable} PRIVATE DawnIIR DawnSerialization DawnUnittest gtest gtest_main)
target_add_dawn_standard_props(${executable})
file(COPY input DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
gtest_discover_tests(${executable} TEST_PREFIX "Dawn::Unit::IIR::" DISCOVERY_TIMEOUT 30)
