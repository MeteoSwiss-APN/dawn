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

yoda_add_library(
  NAME DawnAST
  SOURCES AST.h
          AST.tcc
          ASTExpr.tcc
          ASTExpr.h
          ASTFwd.h
          ASTStmt.tcc
          ASTStmt.h
          ASTStringifier.tcc
          ASTStringifier.h
          ASTUtil.h
          ASTUtil.tcc
          ASTVisitor.tcc
          ASTVisitor.h
          ASTVisitorHelpers.h
  OBJECT
)
