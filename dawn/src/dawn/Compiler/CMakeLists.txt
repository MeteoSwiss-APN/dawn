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

add_library(DawnCompiler
  Driver.h
  Driver.cpp
)

target_add_dawn_standard_props(DawnCompiler)
target_link_libraries(DawnCompiler PUBLIC DawnSupport DawnSIR DawnOptimizer DawnValidator DawnCodeGen DawnSerialization)
