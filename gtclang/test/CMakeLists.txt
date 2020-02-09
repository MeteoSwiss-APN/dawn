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

if(NOT GTCLANG_BUILD_GRIDTOOLS)
  find_package(GridTools 1.0.4)
endif()

if(NOT GridTools_FOUND)
  if(NOT GTCLANG_BUILD_GRIDTOOLS)
    message(STATUS "Fetching GridTools...")
  endif()
  # GridTools uses BUILD_TESTING, cache it and reset after
  set(build_testing ${BUILD_TESTING})
  set(install_tools ${INSTALL_TOOLS})
  set(BUILD_TESTING OFF)
  set(INSTALL_TOOLS OFF)
  set(GT_INSTALL_EXAMPLES OFF)

  FetchContent_Declare(GridTools
    URL https://github.com/GridTools/gridtools/archive/v1.0.4.tar.gz
  )

  FetchContent_GetProperties(GridTools)
  if(NOT GridTools_POPULATED)
    FetchContent_Populate(GridTools)
    add_subdirectory(${gridtools_SOURCE_DIR} ${gridtools_BINARY_DIR})
  endif()

  set(GTCLANG_BUILD_GRIDTOOLS ON CACHE BOOL "Use GridTools from bundle.")
  mark_as_advanced(GTCLANG_BUILD_GRIDTOOLS)

  # Reset here
  set(BUILD_TESTING ${build_testing})
  set(INSTALL_TOOLS ${install_tools})
endif()

add_subdirectory(unit-test)
add_subdirectory(integration-test)
add_subdirectory(atlas-integration-test)
