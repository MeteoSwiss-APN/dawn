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
  NAME GSLCodeGen
  SOURCES CodeGen.h
          CXXUtil.h
          ASTCodeGenCXX.h
          ASTCodeGenCXX.cpp
          ASTCodeGenGTClangStencilBody.h
          ASTCodeGenGTClangStencilBody.cpp
          ASTCodeGenGTClangStencilDesc.h
          ASTCodeGenGTClangStencilDesc.cpp
          GTClangCodeGen.h
          GTClangCodeGen.cpp
          GTClangNaiveCXXCodeGen.h
          GTClangNaiveCXXCodeGen.cpp
          TranslationUnit.h
          TranslationUnit.cpp
)
