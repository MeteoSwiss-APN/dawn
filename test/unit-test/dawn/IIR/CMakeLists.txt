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
  NAME DawnUnittestIIR
  SOURCES TestExtent.cpp
          TestField.cpp
          TestInterval.cpp
          TestIntervalAlgorithms.cpp
          TestIIRNode.cpp
          TestIIRNodeIterator.cpp
          TestMain.cpp
          TestMultiInterval.cpp
          TestStencil.cpp
)

