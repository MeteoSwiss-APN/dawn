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

dawn_protobuf_generate(
  OUT_FILES sir_proto_files
  OUT_INCLUDE_DIRS sir_proto_include_dirs
  PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/SIR.proto
  LANGUAGE cpp
)
include_directories(${sir_proto_include_dirs})

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
          ${sir_proto_files}
  OBJECT
)
