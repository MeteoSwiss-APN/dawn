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

include(yodaExportPackage)
find_package(GridTools CONFIG REQUIRED)

if(GridTools_FOUND)
  set(GTCLANG_HAS_GRIDTOOLS 1)
endif()

yoda_export_package(
  NAME GridTools
  FOUND ${GridTools_FOUND}
  INCLUDE_DIRS ${GridTools_INCLUDE_PATH} 
  VERSION "${GridTools_VERSION}" 
)
