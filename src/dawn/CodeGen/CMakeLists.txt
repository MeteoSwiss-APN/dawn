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

mchbuild_add_library(
  NAME DawnCodeGen
  SOURCES ASTCodeGenCXX.cpp
          ASTCodeGenCXX.h
          CodeGen.h
          CXXUtil.h
          CXXNaive/ASTCodeGenCXXNaiveStencilBody.cpp
          CXXNaive/ASTCodeGenCXXNaiveStencilBody.h
          CXXNaive/ASTCodeGenCXXNaiveStencilDesc.cpp
          CXXNaive/ASTCodeGenCXXNaiveStencilDesc.h
          CXXNaive/ASTCodeGenCXXNaiveStencilFnParamVisitor.cpp
          CXXNaive/ASTCodeGenCXXNaiveStencilFnParamVisitor.h
          CXXNaive/CXXNaiveCodeGen.cpp
          CXXNaive/CXXNaiveCodeGen.h
          GridTools/ASTCodeGenGTStencilBody.cpp
          GridTools/ASTCodeGenGTStencilBody.h
          GridTools/ASTCodeGenGTStencilDesc.cpp
          GridTools/ASTCodeGenGTStencilDesc.h
          GridTools/GTCodeGen.cpp
          GridTools/GTCodeGen.h
          TranslationUnit.cpp
          TranslationUnit.h
  OBJECT
)
