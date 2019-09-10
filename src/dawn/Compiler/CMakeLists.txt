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

yoda_add_library(
  NAME DawnCompiler
  SOURCES DawnCompiler.h
          DawnCompiler.cpp
          Options.h
          Options.inc
          test_proto.h
          test_proto.cpp
  OBJECT
)

target_include_directories(DawnCompilerObjects PUBLIC $<TARGET_PROPERTY:DawnSIRStatic,INCLUDE_DIRECTORIES>)

