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

add_executable(DawnUnittestAST TestMain.cpp TestOffset.cpp)
target_link_libraries(DawnUnittestAST PRIVATE DawnSIR DawnUnittest gtest)
target_add_dawn_standard_props(DawnUnittestAST)
add_test(NAME DawnUnittestAST
  COMMAND DawnUnittestAST
)
