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

add_library(DawnSerialization
  ASTSerializer.h
  ASTSerializer.cpp
  IIRSerializer.h
  IIRSerializer.cpp
  SIRSerializer.h
  SIRSerializer.cpp
)

target_add_dawn_standard_props(DawnSerialization)
target_link_libraries(DawnSerialization
  PUBLIC DawnSupport DawnAST DawnSIR DawnIIR
  PRIVATE protobuf::libprotobuf
)
