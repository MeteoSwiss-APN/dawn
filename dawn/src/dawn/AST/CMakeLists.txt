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
  LocationType.h
  GridType.h
  GridType.cpp
  Offsets.h
  Offsets.cpp
  Tags.h
)

target_add_dawn_standard_props(DawnAST)
target_link_libraries(DawnAST PUBLIC DawnSupport)
