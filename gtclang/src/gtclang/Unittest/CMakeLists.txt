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

set(GTCLANG_EXECUTABLE "${PROJECT_BINARY_DIR}/bin/gtclang")
set(GTCLANG_UNITTEST_DATAPATH ${PROJECT_BINARY_DIR}/test/unit-test)
set(GTCLANG_INTEGRATIONTEST_DATAPATH ${PROJECT_BINARY_DIR}/test/integration-test)
set(GTCLANG_UNITTEST_INCLUDES ${PROJECT_SOURCE_DIR}/src)

# This two-step hack is necessary because there are both variables and generator
# expressions necessary to generate config.py
configure_file(${config_in} ${CMAKE_CURRENT_BINARY_DIR}/.tmp)
file(GENERATE OUTPUT ${config_out} INPUT ${CMAKE_CURRENT_BINARY_DIR}/.tmp)

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
target_link_libraries(GTClangUnittest PUBLIC Dawn::Dawn LLVM::LLVM Clang gtest)
