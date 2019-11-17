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

add_library(DawnUnittest OBJECT
  IIRBuilder.h
  IIRBuilder.cpp
  UnittestLogger.cpp
  UnittestLogger.h
)

target_include_directories(DawnUnittest
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
)
target_compile_features(DawnUnittest PUBLIC cxx_std_17)
target_compile_options(DawnUnittest PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(DawnUnittest PROPERTIES ${DAWN_TARGET_PROPERTIES})
target_link_libraries(DawnUnittest PUBLIC DawnSupport DawnIIR DawnOptimizer DawnCodeGen)
