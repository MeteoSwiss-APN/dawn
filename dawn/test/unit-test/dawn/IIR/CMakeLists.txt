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

set(executable ${PROJECT_NAME}UnittestIIR)
add_executable(${executable}
  TestExtent.cpp
  TestField.cpp
  TestInterval.cpp
  TestIntervalAlgorithms.cpp
  TestIIRNode.cpp
  TestIIRNodeIterator.cpp
  TestMain.cpp
  TestMultiInterval.cpp
  TestStencil.cpp
  TestIIRSerializer.cpp
)
target_link_libraries(${executable} PRIVATE DawnIIR DawnSerialization DawnUnittest gtest)
target_add_dawn_standard_props(${executable})
add_test(NAME ${executable} COMMAND ${executable})
