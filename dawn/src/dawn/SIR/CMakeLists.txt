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

# Defines dawn_sir_proto_files
include(proto/DawnSIR.cmake)

include(DawnProtobufGenerate)
dawn_protobuf_generate(
  OUT_FILES sir_proto_cpp_files
  WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
  PROTOS ${dawn_sir_proto_files}
  PACKG SIR
  LANGUAGE cpp
)

add_library(DawnSIRProto OBJECT ${sir_proto_cpp_files})
target_compile_features(DawnSIRProto PUBLIC cxx_std_11)
set_target_properties(DawnSIRProto PROPERTIES ${DAWN_TARGET_PROPERTIES})

# Generated files need to be compiled with the protobuf headers
target_include_directories(DawnSIRProto
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(DawnSIRProto PUBLIC protobuf::libprotobuf)

# Builds one library for both SIR and AST
add_library(DawnSIR
  AST.h
  ASTExpr.h
  ASTFwd.h
  ASTStmt.h
  ASTStmt.cpp
  ASTStringifier.h
  ASTUtil.h
  ASTUtil.cpp
  ASTVisitor.h
  SIR.cpp
  SIR.h
  $<TARGET_OBJECTS:DawnSIRProto>
)

target_add_dawn_standard_props(DawnSIR)
target_link_libraries(DawnSIR PUBLIC DawnSupport DawnAST DawnSIRProto)
