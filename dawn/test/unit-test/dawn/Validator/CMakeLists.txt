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

set(executable ${PROJECT_NAME}UnittestValidator)
add_executable(${executable}
  TestMain.cpp  
  TestTypeChecker.cpp
)
target_link_libraries(${executable} DawnValidator DawnUnittest gtest)
target_add_dawn_standard_props(${executable})
gtest_discover_tests(${executable} TEST_PREFIX "${executable}::")