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

add_library(DawnUnittest
  CodeDumper.cpp
  CodeDumper.h
  IIRBuilder.h
  IIRBuilder.cpp
  UnittestLogger.cpp
  UnittestLogger.h
)

target_add_dawn_standard_props(DawnUnittest)
target_link_libraries(DawnUnittest PUBLIC DawnSupport DawnIIR DawnOptimizer DawnValidator DawnCodeGen)
