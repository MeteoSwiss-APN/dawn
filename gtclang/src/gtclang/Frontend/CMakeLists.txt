##===------------------------------------------------------------------------------*- CMake -*-===##
##                         _       _                   
##                        | |     | |                  
##                    __ _| |_ ___| | __ _ _ __   __ _ 
##                   / _` | __/ __| |/ _` | '_ \ / _` |
##                  | (_| | || (__| | (_| | | | | (_| |
##                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
##                    __/ |                       __/ |
##                   |___/                       |___/ 
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

add_library(GTClangFrontend
  ClangASTExprResolver.h
  ClangASTExprResolver.cpp
  ClangASTStmtResolver.cpp
  ClangASTStmtResolver.h
  ClangFormat.cpp
  ClangFormat.h
  Diagnostics.cpp
  Diagnostics.h
  DiagnosticsKind.inc
  GlobalVariableParser.cpp
  GlobalVariableParser.h
  GTClangASTAction.cpp
  GTClangASTAction.h
  GTClangASTConsumer.cpp
  GTClangASTConsumer.h
  GTClangASTVisitor.cpp
  GTClangASTVisitor.h
  GTClangContext.cpp
  GTClangContext.h
  GTClangIncludeChecker.cpp
  GTClangIncludeChecker.h
  GTClangPreprocessorAction.cpp
  GTClangPreprocessorAction.h
  StencilParser.cpp
  StencilParser.h
)

target_add_gtclang_standard_props(GTClangFrontend)
target_link_libraries(GTClangFrontend
  PUBLIC GTClangSupport Dawn::Dawn Clang::Clang LLVM::LLVM
)
