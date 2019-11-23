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

add_library(${PROJECT_NAME}C
  Dawn.h
  Compiler.cpp
  Compiler.h
  ErrorHandling.cpp
  ErrorHandling.h
  Options.cpp
  Options.h
  TranslationUnit.cpp
  TranslationUnit.h
  Types.h
  util/Allocate.h
  util/CompilerWrapper.h
  util/OptionsWrapper.cpp
  util/OptionsWrapper.h
  util/TranslationUnitWrapper.h
)

target_add_dawn_standard_props(${PROJECT_NAME}C)
target_link_libraries(${PROJECT_NAME}C PUBLIC ${PROJECT_NAME})
