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

set(executable ${PROJECT_NAME}UnittestSupport)
add_executable(${executable}
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
target_link_libraries(${executable} DawnSupport DawnUnittest gtest)
target_add_dawn_standard_props(${executable})
gtest_discover_tests(${executable} TEST_PREFIX "${executable}::")
