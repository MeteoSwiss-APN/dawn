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
  OBJECT
)

yoda_combine_libraries(
  NAME DawnAST
  OBJECTS DawnASTObjects
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  DEPENDS DawnSupportStatic DawnSIRStatic
  VERSION ${DAWN_VERSION}
  EXPORT_GROUP DawnTargets
)
