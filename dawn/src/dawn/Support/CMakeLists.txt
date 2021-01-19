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

configure_file(Config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/Config.h)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/Config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dawn/Support
)

add_library(DawnSupport
  ${CMAKE_CURRENT_BINARY_DIR}/Config.h
  ${PROJECT_SOURCE_DIR}/src/dawn/Dawn.h
  AlignOf.h
  Array.cpp
  Array.h
  ArrayRef.h
  Assert.cpp
  Assert.h
  Casting.h
  Compiler.h
  ComparisonHelpers.h
  ContainerUtils.h
  DoubleSidedMap.h
  EditDistance.h
  Exception.h
  Exception.cpp
  Format.h
  HashCombine.h
  IndexGenerator.cpp
  IndexGenerator.h
  IndexRange.h
  Iterator.h
  Json.h
  Logger.cpp
  Logger.h
  NonCopyable.h
  Printing.h
  RemoveIf.hpp
  SourceLocation.cpp
  SourceLocation.h
  STLExtras.h
  StringSwitch.h
  StringUtil.cpp
  StringUtil.h
  Type.cpp
  Type.h
  TypeTraits.h
  UIDGenerator.cpp
  UIDGenerator.h
  Unreachable.cpp
  Unreachable.h
)

target_add_dawn_standard_props(DawnSupport)
target_include_directories(DawnSupport SYSTEM
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/External>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dawn/Support/External>
)
