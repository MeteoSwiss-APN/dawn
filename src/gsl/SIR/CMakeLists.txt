##===------------------------------------------------------------------------------*- CMake -*-===##
##                                 ____ ____  _     
##                                / ___/ ___|| |    
##                               | |  _\___ \| |    
##                               | |_| |___) | |___ 
##                                \____|____/|_____| - Generic Stencil Language
##                  
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

gsl_add_library(
  NAME GSLSIR
  SOURCES AST.h  
          AST.cpp
          ASTFwd.h
          ASTVisitor.h
          ASTVisitor.cpp  
          ASTStmt.h
          ASTStmt.cpp          
          ASTExpr.h
          ASTExpr.cpp          
          ASTStringifier.h
          ASTStringifier.cpp
          ASTUtil.h
          ASTUtil.cpp
          SIR.h
          SIR.cpp
          SIRSerializerJSON.h          
          SIRSerializerJSON.cpp
          Statement.h
)
