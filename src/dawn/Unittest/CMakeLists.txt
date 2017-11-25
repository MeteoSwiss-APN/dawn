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

mchbuild_add_library(
  NAME DawnUnittest
  SOURCES ASTSimplifier.cpp
          ASTSimplifier.h
          PrintAllExpressionTypes.cpp
          PrintAllExpressionTypes.h
          UnittestLogger.cpp
          UnittestLogger.h
  ARCHIVE
)
