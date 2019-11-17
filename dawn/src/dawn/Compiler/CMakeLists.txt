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

add_library(DawnCompiler OBJECT
  DawnCompiler.h
  DawnCompiler.cpp
  Options.h
  Options.inc
)

target_include_directories(DawnCompiler
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
)
target_compile_features(DawnCompiler PUBLIC cxx_std_17)
target_compile_options(DawnCompiler PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(DawnCompiler PROPERTIES ${DAWN_TARGET_PROPERTIES})
target_link_libraries(DawnCompiler PUBLIC DawnSupport DawnSIR DawnOptimizer DawnCodeGen DawnSerialization)
