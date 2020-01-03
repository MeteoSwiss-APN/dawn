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

set(protos SIR/SIR.proto SIR/statements.proto SIR/enums.proto)

include(DawnProtobufGenerate)
dawn_protobuf_generate(
  OUT_FILES sir_proto_cpp_files
  OUT_INCLUDE_DIRS sir_proto_include_dirs
  WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
  PROTOS ${protos}
  PACKG SIR
  LANGUAGE cpp
)

if(ENABLE_PYTHON)
  # Generate python proto files
  include(DawnProtobufGenerate)
  dawn_protobuf_generate(
    OUT_FILES sir_proto_python_files
    PROTOS ${protos}
    WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PACKG SIR
    LANGUAGE python
  )

  set(output_dir ${PROJECT_SOURCE_DIR}/src/dawn4py/serialization/SIR)

  string(REGEX REPLACE ${CMAKE_CURRENT_BINARY_DIR} ${output_dir}
    sir_proto_python_build_files "${sir_proto_python_files}")

  add_custom_command(OUTPUT ${sir_proto_python_build_files}
    COMMAND ${CMAKE_COMMAND} -E copy ${sir_proto_python_files} ${output_dir}
    DEPENDS ${sir_proto_python_files}
  )
endif()

add_library(DawnSIRProto OBJECT ${sir_proto_cpp_files} ${sir_proto_python_build_files})
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
  $<TARGET_OBJECTS:DawnAST>
)

target_add_dawn_standard_props(DawnSIR)
target_link_libraries(DawnSIR PUBLIC DawnSupport DawnAST DawnSIRProto)
