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

add_library(DawnSerialization OBJECT
  ASTSerializer.h
  ASTSerializer.cpp
  IIRSerializer.h
  IIRSerializer.cpp
  SIRSerializer.h
  SIRSerializer.cpp
)

target_include_directories(DawnSerialization
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_compile_features(DawnSerialization PUBLIC cxx_std_17)
target_compile_options(DawnSerialization PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(DawnSerialization PROPERTIES ${DAWN_TARGET_PROPERTIES})
target_link_libraries(DawnSerialization PUBLIC DawnSupport DawnAST DawnSIR DawnIIR DawnOptimizer)
