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

  mchbuild_add_unittest(
    NAME DawnUnittestOptimizerPasses
    SOURCES 
          TestMain.cpp
          TestPassComputeStageExtents.cpp
          TestPassSetBoundaryCondition.cpp
          TestFieldAccessIntervals.cpp
          TestTemporaryToFunction.cpp
    DEPENDS DawnUnittestStatic DawnStatic DawnCStatic ${DAWN_EXTERNAL_LIBRARIES} gtest
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/unittest
    GTEST_ARGS "${CMAKE_CURRENT_LIST_DIR}" "--gtest_color=yes"
  )

target_include_directories(DawnUnittestOptimizerPasses PUBLIC "${CMAKE_SOURCE_DIR}")

