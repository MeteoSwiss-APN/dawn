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

gtclang_add_library(
  NAME GTClangFrontend
  SOURCES ClangASTExprResolver.h
          ClangASTExprResolver.cpp
          ClangASTStmtResolver.h
          ClangASTStmtResolver.cpp
          ClangFormat.h
          ClangFormat.cpp
          Diagnostics.h
          Diagnostics.cpp
          DiagnosticsKind.inc
          GlobalVariableParser.h
          GlobalVariableParser.cpp
          GTClangASTAction.h
          GTClangASTAction.cpp
          GTClangASTConsumer.h
          GTClangASTConsumer.cpp
          GTClangASTVisitor.h
          GTClangASTVisitor.cpp
          GTClangContext.h
          GTClangContext.cpp
          GTClangPreprocessorAction.h
          GTClangPreprocessorAction.cpp
          StencilParser.h
          StencilParser.cpp
)
