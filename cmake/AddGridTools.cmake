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

find_package(GridTools)

if(GRIDTOOLS_FOUND)
  include_directories(SYSTEM ${GRIDTOOLS_INCLUDE_DIRS})
  set(GRIDTOOLS_LIBRARIES)

  gtclang_export_package_variable(
    GRIDTOOLS 
    ${GRIDTOOLS_FOUND} 
    "gridtools: found" 
    ${GRIDTOOLS_LIBRARIES}
)
endif()

