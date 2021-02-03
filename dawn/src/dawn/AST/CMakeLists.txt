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

# Defines dawn_ast_proto_files
include(proto/DawnAST.cmake)

include(DawnProtobufGenerate)
dawn_protobuf_generate(
  OUT_FILES ast_proto_cpp_files
  WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
  PROTOS ${dawn_ast_proto_files}
  PACKG AST
  LANGUAGE cpp
)

set(ast_proto_header_files ${ast_proto_cpp_files})
list(FILTER ast_proto_header_files INCLUDE REGEX ".+\\.h?h$")
install(FILES ${ast_proto_header_files} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dawn/AST/AST)

add_library(DawnASTProto OBJECT ${ast_proto_cpp_files})
target_compile_features(DawnASTProto PUBLIC cxx_std_11)
set_target_properties(DawnASTProto PROPERTIES ${DAWN_TARGET_PROPERTIES})

# Generated files need to be compiled with the protobuf headers
target_include_directories(DawnASTProto
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        )

# ... and linked to protobuf
target_link_libraries(DawnASTProto PUBLIC protobuf::libprotobuf)

add_library(DawnAST OBJECT
  AST.h
  AST.cpp
  ASTExpr.cpp
  ASTExpr.h
  ASTFwd.h
  ASTStmt.cpp
  ASTStmt.h
  ASTStringifier.cpp
  ASTStringifier.h
  ASTUtil.h
  ASTUtil.cpp
  ASTVisitor.cpp
  ASTVisitor.h
  ASTVisitorHelpers.h
  Attr.h
  FieldDimension.h
  FieldDimension.cpp
  GridType.h
  GridType.cpp
  Interval.h
  Interval.cpp
  LocationType.h
  Offsets.h
  Offsets.cpp
  Tags.h
  Value.cpp
  Value.h
  $<TARGET_OBJECTS:DawnASTProto>
)

target_add_dawn_standard_props(DawnAST)
target_link_libraries(DawnAST PUBLIC DawnSupport DawnASTProto)

# The include path below is necessary for the C++ proto headers
target_include_directories(DawnAST
        PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dawn/AST>
        )
