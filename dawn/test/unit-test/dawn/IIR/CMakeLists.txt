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

add_executable(DawnUnittestIIR
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
target_link_libraries(DawnUnittestIIR PRIVATE DawnIIR DawnSerialization DawnUnittest gtest)
target_add_dawn_standard_props(DawnUnittestIIR)
add_test(NAME DawnUnittestIIR
  COMMAND DawnUnittestIIR
)
