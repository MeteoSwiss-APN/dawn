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

add_executable(DawnUnittestSIR
  TestMain.cpp
  TestAST.cpp
  TestASTVisitor.cpp
  TestSIR.cpp
  TestSIRSerializer.cpp
)
target_link_libraries(DawnUnittestSIR Dawn gtest)
target_compile_options(DawnUnittestSIR PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(DawnUnittestSIR PROPERTIES ${DAWN_TARGET_PROPERTIES})
add_test(NAME DawnUnittestSIR
  COMMAND DawnUnittestSIR
)
