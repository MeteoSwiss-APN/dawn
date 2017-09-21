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

set(GTCLANG_EXECUTABLE ${GTCLANG_EXECUTABLE})
set(GTCLANG_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/src")
find_package(GTClang REQUIRED)
include_directories(SYSTEM ${GTCLANG_INCLUDE_DIRS})
set(GTCLANG_LIBRARIES)

gtclang_export_package_variable(
  GTCLANG 
  ${GTCLANG_FOUND} 
  "GTClang: found" 
  ${GTCLANG_LIBRARIES}
)
