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

add_executable(DawnCUnittest
  TestMain.cpp
  TestOptions.cpp
  TestCompiler.cpp
)

target_add_dawn_standard_props(DawnCUnittest)
target_link_libraries(DawnCUnittest DawnC DawnUnittest gtest)
