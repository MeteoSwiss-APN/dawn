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
  NAME DawnSerializer
  SOURCES ASTSerializer.h
          ASTSerializer.cpp
          IIRSerializer.h
          IIRSerializer.cpp
          SIRSerializer.h
          SIRSerializer.cpp
  OBJECT
)

add_dependencies(DawnSerializerObjects IIR_Proto_Generated)
target_include_directories(DawnSerializerObjects PUBLIC $<TARGET_PROPERTY:DawnSIRObjects,INCLUDE_DIRECTORIES>)
