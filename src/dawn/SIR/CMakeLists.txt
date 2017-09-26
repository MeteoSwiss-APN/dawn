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
          SIRSerializerJSON.cpp
          SIRSerializerJSON.h          
          Statement.h
  OBJECT
)
