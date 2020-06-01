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

add_library(DawnCodeGen
  ASTCodeGenCXX.cpp
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
  CXXNaive-ico/ASTStencilBody.cpp
  CXXNaive-ico/ASTStencilBody.h
  CXXNaive-ico/ASTStencilDesc.cpp
  CXXNaive-ico/ASTStencilDesc.h
  CXXNaive-ico/ASTStencilFunctionParamVisitor.cpp
  CXXNaive-ico/ASTStencilFunctionParamVisitor.h
  CXXNaive-ico/CXXNaiveCodeGen.cpp
  CXXNaive-ico/CXXNaiveCodeGen.h
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
  Cuda/MSCodeGen.cpp
  Cuda/MSCodeGen.h
  Cuda-ico/ASTStencilBody.h
  Cuda-ico/ASTStencilBody.cpp
  Cuda-ico/CudaIcoCodeGen.h
  Cuda-ico/CudaIcoCodeGen.cpp
  Cuda-ico/LocToStringUtils.h
  Cuda-ico/LocToStringUtils.cpp
  Driver.cpp
  Driver.h
  GridTools/ASTStencilBody.cpp
  GridTools/ASTStencilBody.h
  GridTools/ASTStencilDesc.cpp
  GridTools/ASTStencilDesc.h
  GridTools/CodeGenUtils.cpp
  GridTools/CodeGenUtils.h
  GridTools/GTCodeGen.cpp
  GridTools/GTCodeGen.h
  Options.h
  Options.inc
  StencilFunctionAsBCGenerator.cpp
  StencilFunctionAsBCGenerator.h
  TranslationUnit.cpp
  TranslationUnit.h
)

target_add_dawn_standard_props(DawnCodeGen)
target_link_libraries(DawnCodeGen PUBLIC DawnSupport DawnIIR)
