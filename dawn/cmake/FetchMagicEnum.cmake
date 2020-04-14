##===------------------------------------------------------------------------------*- CMake -*-===##
##                          _
##                         | |
##                       __| | __ ___      ___ ___
##                      / _` |/ _` \ \ /\ / / '_  |
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT).
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

if(NOT TARGET magic_enum::magic_enum)
  if(NOT INCLUDE_MAGIC_ENUM)
    message(STATUS "Fetching magic_enum...")
  endif()
  set(MAGIC_ENUM_OPT_BUILD_EXAMPLES OFF)
  set(MAGIC_ENUM_OPT_BUILD_TESTS OFF)

  FetchContent_Declare(magic_enum
    URL https://github.com/Neargye/magic_enum/archive/v0.6.5.tar.gz
  )

  FetchContent_GetProperties(magic_enum)
  if(NOT magic_enum_POPULATED)
    FetchContent_Populate(magic_enum)
    add_subdirectory(${magic_enum_SOURCE_DIR} ${magic_enum_BINARY_DIR})
  endif()

  set(INCLUDE_MAGIC_ENUM ON CACHE BOOL "Fetched magic_enum.")
  mark_as_advanced(INCLUDE_MAGIC_ENUM)
endif()
