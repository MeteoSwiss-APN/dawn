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
target_link_libraries(DawnUnittestAST Dawn gtest)
target_compile_options(DawnUnittestAST PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(DawnUnittestAST PROPERTIES ${DAWN_TARGET_PROPERTIES})
add_test(NAME DawnUnittestAST
  COMMAND DawnUnittestAST
)
