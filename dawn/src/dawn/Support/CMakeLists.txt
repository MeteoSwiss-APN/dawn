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

add_library(DawnSupport OBJECT
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
  DiagnosticsEngine.cpp
  DiagnosticsEngine.h
  DiagnosticsMessage.h
  DiagnosticsQueue.cpp
  DiagnosticsQueue.h
  DoubleSidedMap.h
  EditDistance.h
  FileUtil.cpp
  FileUtil.h
  Format.h
  HashCombine.h
  IndexGenerator.cpp
  IndexGenerator.h
  IndexRange.h
  Json.h
  Logging.cpp
  Logging.h
  MathExtras.h
  NonCopyable.h
  Printing.h
  RemoveIf.hpp
  SmallString.h
  SmallVector.cpp
  SmallVector.h
  SourceLocation.cpp
  SourceLocation.h
  STLExtras.h
  StringRef.cpp
  StringRef.h
  StringSwitch.h
  StringUtil.cpp
  StringUtil.h
  Twine.cpp
  Twine.h
  Type.cpp
  Type.h
  TypeTraits.h
  UIDGenerator.cpp
  UIDGenerator.h
  Unreachable.cpp
  Unreachable.h
)

target_include_directories(DawnSupport
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/External>
)
target_compile_features(DawnSupport PUBLIC cxx_std_17)
target_compile_options(DawnSupport PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(DawnSupport PROPERTIES ${DAWN_TARGET_PROPERTIES})
