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

if(NOT USE_BUNDLED_GRIDTOOLS)
  find_package(GridTools 1.0.3)
endif()
if(NOT GridTools_FOUND)
  # GridTools uses BUILD_TESTING, cache it and reset after
  set(build_testing ${BUILD_TESTING})
  set(BUILD_TESTING OFF)
  set(INSTALL_TOOLS OFF)

  # NOTE: Need CMake fixes to v1.0.3
  FetchContent_Declare(GridTools
    GIT_REPOSITORY https://github.com/jdahm/gridtools.git
    GIT_TAG v1.0.3/fix-cmake-project
  )

  FetchContent_GetProperties(GridTools)
  if(NOT GridTools_POPULATED)
    FetchContent_Populate(GridTools)
    add_subdirectory(${gridtools_SOURCE_DIR} ${gridtools_BINARY_DIR})
  endif()

  set(USE_BUNDLED_GRIDTOOLS ON CACHE BOOL "Use GridTools from bundle.")
  mark_as_advanced(USE_BUNDLED_GRIDTOOLS)

  # Reset here
  set(BUILD_TESTING ${build_testing})
endif()

add_subdirectory(unit-test)
add_subdirectory(integration-test)
