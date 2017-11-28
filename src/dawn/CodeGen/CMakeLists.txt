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
          ASTCodeGenGTClangStencilBody.cpp
          ASTCodeGenGTClangStencilBody.h
          ASTCodeGenGTClangStencilDesc.cpp
          ASTCodeGenGTClangStencilDesc.h
          CodeGen.h
          CXXUtil.h
          GTClangCodeGen.cpp
          GTClangCodeGen.h
          GTClangNaiveCXXCodeGen.cpp
          GTClangNaiveCXXCodeGen.h
          TranslationUnit.cpp
          TranslationUnit.h
  OBJECT
)
