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
  Offsets.h
  Offsets.cpp
)

target_include_directories(DawnAST
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
)
target_compile_features(DawnAST PUBLIC cxx_std_17)
target_compile_options(DawnAST PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(DawnAST PROPERTIES ${DAWN_TARGET_PROPERTIES})
target_link_libraries(DawnAST PUBLIC DawnSupport)
