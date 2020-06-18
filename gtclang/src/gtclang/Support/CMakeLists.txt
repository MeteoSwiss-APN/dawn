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

set(config_in "${CMAKE_CURRENT_SOURCE_DIR}/Config.h.cmake")
set(config_out "${CMAKE_CURRENT_BINARY_DIR}/Config.h")

configure_file(${config_in} ${config_out})

add_library(GTClangSupport
  ${CMAKE_CURRENT_BINARY_DIR}/Config.h
  FileUtil.cpp
  FileUtil.h
  Logger.cpp
  Logger.h
  StringUtil.cpp
  StringUtil.h
  ASTUtils.cpp
  ASTUtils.h
  MathExtras.h
  SmallString.h
  SmallVector.cpp
  SmallVector.h
  StringRef.cpp
  StringRef.h
)

target_add_gtclang_standard_props(GTClangSupport)
target_link_libraries(GTClangSupport PUBLIC Dawn::DawnSupport Clang::Clang LLVM::LLVM)
