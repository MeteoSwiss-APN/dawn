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

set(GTCLANG_EXECUTABLE "${PROJECT_BINARY_DIR}/bin/gtclang")
set(GTCLANG_UNITTEST_DATAPATH ${PROJECT_BINARY_DIR}/test/unit-test)
set(GTCLANG_INTEGRATIONTEST_DATAPATH ${PROJECT_BINARY_DIR}/test/integration-test)
set(GTCLANG_UNITTEST_INCLUDES ${PROJECT_SOURCE_DIR}/src)
configure_file(Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/Config.h @ONLY)

add_library(GTClangUnittest
  ${CMAKE_CURRENT_BINARY_DIR}/Config.h
  FileManager.cpp
  FileManager.h
  FlagManager.cpp
  FlagManager.h
  GTClang.cpp
  GTClang.h
  UnittestEnvironment.cpp
  UnittestEnvironment.h
  UnittestStmtSimplifier.h
  UnittestStmtSimplifier.cpp
  ParsingComparison.h
  ParsingComparison.cpp
  ParsedString.h
  ParsedString.cpp
)

target_add_gtclang_standard_props(GTClangUnittest)
target_link_libraries(GTClangUnittest PUBLIC GTClangSupport GTClangDriver Dawn::Dawn LLVM::LLVM Clang gtest)
