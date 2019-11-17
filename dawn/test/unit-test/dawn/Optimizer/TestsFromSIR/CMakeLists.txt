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


add_executable(DawnUnittestOptimizerFromSIR
  TestMain.cpp
  TestComputeMaximumExtent.cpp
  TestMultiStage.cpp
  TestStage.cpp
)
target_link_libraries(DawnUnittestOptimizerFromSIR Dawn gtest)
target_compile_options(DawnUnittestOptimizer PRIVATE ${DAWN_CXX_FLAGS})
target_include_directories(DawnUnittestOptimizerFromSIR PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(DawnUnittestOptimizerFromSIR PROPERTIES ${DAWN_TARGET_PROPERTIES})
add_test(NAME DawnUnittestOptimizerFromSIR
  COMMAND DawnUnittestOptimizerFromSIR "${CMAKE_CURRENT_LIST_DIR}/../Passes"
)
