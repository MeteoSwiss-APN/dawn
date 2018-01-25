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

dawn_add_unittest_impl(
  NAME DawnUnittestOptimizer
  SOURCES TestColoringAlgorithm.cpp
          TestExtent.cpp  
          TestGraph.cpp
          TestIsDAGAlgorithm.cpp          
          TestPartitionAlgorithm.cpp
          TestMain.cpp
          TestInterval.cpp
          TestSCCAlgorithm.cpp
          TestStencil.cpp
)

add_subdirectory(Passes)
