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
  NAME DawnCodeGen
  SOURCES ASTCodeGenCXX.cpp
          ASTCodeGenCXX.h
          CodeGen.h
          CodeGen.cpp
          CodeGenProperties.cpp
          CodeGenProperties.h
          CXXUtil.h
          CXXNaive/ASTStencilBody.cpp
          CXXNaive/ASTStencilBody.h
          CXXNaive/ASTStencilDesc.cpp
          CXXNaive/ASTStencilDesc.h
          CXXNaive/ASTStencilFunctionParamVisitor.cpp
          CXXNaive/ASTStencilFunctionParamVisitor.h
          CXXNaive/CXXNaiveCodeGen.cpp
          CXXNaive/CXXNaiveCodeGen.h
          Cuda/CacheProperties.cpp
          Cuda/CacheProperties.h
          Cuda/CodeGeneratorHelper.cpp
          Cuda/CodeGeneratorHelper.h
          Cuda/CudaCodeGen.cpp
          Cuda/CudaCodeGen.h
          Cuda/ASTStencilBody.cpp
          Cuda/ASTStencilBody.h
          Cuda/ASTStencilDesc.cpp
          Cuda/ASTStencilDesc.h
          Cuda/ASTStencilFunctionParamVisitor.cpp
          Cuda/ASTStencilFunctionParamVisitor.h
          GridTools/ASTStencilBody.cpp
          GridTools/ASTStencilBody.h
          GridTools/ASTStencilDesc.cpp
          GridTools/ASTStencilDesc.h
          GridTools/GTCodeGen.cpp
          GridTools/GTCodeGen.h
          StencilFunctionAsBCGenerator.cpp
          StencilFunctionAsBCGenerator.h
          TranslationUnit.cpp
          TranslationUnit.h
  OBJECT
)
