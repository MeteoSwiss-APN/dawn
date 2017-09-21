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

## gtclang_include_guard
## -----------------
##
## Prevent frequently-included CMake files from being re-parsed multiple times.
##
macro(gtclang_include_guard)
  if(DEFINED "__GTCLANG_INCLUDE_GUARD_${CMAKE_CURRENT_LIST_FILE}")
    return()
  endif(DEFINED "__GTCLANG_INCLUDE_GUARD_${CMAKE_CURRENT_LIST_FILE}")

  set("__GTCLANG_INCLUDE_GUARD_${CMAKE_CURRENT_LIST_FILE}" 1)
endmacro()
