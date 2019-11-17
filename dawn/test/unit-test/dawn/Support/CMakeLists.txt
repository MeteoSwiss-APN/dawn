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

add_executable(DawnUnittestSupport
  TestMain.cpp
  TestSmallVector.cpp
  TestStringRef.cpp
  TestArrayRef.cpp
  TestIndexRange.cpp
  TestMain.cpp
  TestRemoveIf.cpp
  TestRangeToString.cpp
  TestType.cpp
)
target_link_libraries(DawnUnittestSupport Dawn gtest)
target_compile_options(DawnUnittestSupport PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(DawnUnittestSupport PROPERTIES ${DAWN_TARGET_PROPERTIES})
add_test(NAME DawnUnittestSupport
  COMMAND DawnUnittestSupport
)
