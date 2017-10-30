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

include(DawnProtobufGenerate)

# Genreate C++ proto files
dawn_protobuf_generate(
  OUT_FILES sir_proto_cpp_files
  OUT_INCLUDE_DIRS sir_proto_include_dirs
  PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/SIR.proto
  LANGUAGE cpp
)
include_directories(SYSTEM ${sir_proto_include_dirs})

dawn_add_library(
  NAME DawnSIR
  SOURCES AST.h
          AST.cpp
          ASTExpr.cpp
          ASTExpr.h
          ASTFwd.h
          ASTStmt.cpp
          ASTStmt.h
          ASTStringifier.cpp
          ASTStringifier.h
          ASTUtil.cpp
          ASTUtil.h
          ASTVisitor.cpp
          ASTVisitor.h
          SIR.cpp
          SIR.h
          SIR.proto
          SIRSerializerJSON.cpp
          SIRSerializerJSON.h
          Statement.h
          ${sir_proto_cpp_files}
  OBJECT
)

# Genreate python proto files
dawn_protobuf_generate(
  OUT_FILES sir_proto_python_files
  PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/SIR.proto
  LANGUAGE python
)
add_custom_target(DawnPythonSIRProto ALL SOURCES ${sir_proto_python_files})

install(FILES ${sir_proto_python_files} DESTINATION python/dawn/protobuf)
