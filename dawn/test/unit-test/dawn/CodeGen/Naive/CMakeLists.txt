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

set(executable ${PROJECT_NAME}UnittestCodeGenNaive)
add_executable(${executable} TestCodeGenNaive.cpp TestMain.cpp)
target_add_dawn_standard_props(${executable})
target_link_libraries(${executable} DawnOptimizer DawnCompiler DawnUnittest gtest)
gtest_discover_tests(${executable} TEST_PREFIX "Dawn::Unit::CodeGen::Naive::" DISCOVERY_TIMEOUT 30)
file(COPY reference DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

