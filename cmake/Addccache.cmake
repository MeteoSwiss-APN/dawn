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

find_package(ccache)

dawn_export_package(
  NAME ccache
  FOUND ${CCACHE_FOUND}
  EXECUTABLE ${CCACHE_EXECUTABLE}
)

if(CCACHE_FOUND)
  set(GTCLANG_HAS_CCACHE 1)
  if(GTCLANG_USE_CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_EXECUTABLE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_EXECUTABLE})
  endif()
else()
  set(GTCLANG_HAS_CCACHE 0)
endif()
