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

set(executable ${PROJECT_NAME}UnittestSIR)
add_executable(${executable}
  TestMain.cpp
  TestAST.cpp
  TestASTVisitor.cpp
  TestSIR.cpp
  TestSIRSerializer.cpp
)
target_link_libraries(${executable} PRIVATE DawnSIR DawnSerialization DawnUnittest gtest)
target_add_dawn_standard_props(${executable})
add_test(NAME ${executable} COMMAND ${executable})
