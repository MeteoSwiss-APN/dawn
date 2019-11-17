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
add_library(Dawn::DawnC ALIAS ${PROJECT_NAME}C)

target_include_directories(${PROJECT_NAME}C
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_compile_features(${PROJECT_NAME}C PUBLIC cxx_std_17)
target_compile_options(${PROJECT_NAME}C PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(${PROJECT_NAME}C PROPERTIES ${DAWN_TARGET_PROPERTIES})
target_link_libraries(${PROJECT_NAME}C PUBLIC Dawn)
