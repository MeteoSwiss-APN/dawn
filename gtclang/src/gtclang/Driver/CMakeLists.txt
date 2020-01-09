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

add_library(GTClangDriver
  CompilerInstance.cpp
  CompilerInstance.h
  Driver.cpp
  Driver.h
  Options.h
  Options.inc
  OptionsParser.cpp
  OptionsParser.h
)

target_add_gtclang_standard_props(GTClangDriver)
target_link_libraries(GTClangDriver
  PUBLIC GTClangFrontend GTClangSupport Dawn::Dawn Clang::Clang LLVM::LLVM
)
