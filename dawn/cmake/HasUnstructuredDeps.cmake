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

if (NOT DEFINED HAS_UNSTRUCTURED_DEPS)

  set(HAS_UNSTRUCTURED_DEPS OFF CACHE BOOL "True if Dawn has detected necessary dependencies for supporting unstructured grids.")
  mark_as_advanced(HAS_UNSTRUCTURED_DEPS)

  if(DAWN_REQUIRE_UNSTRUCTURED_TESTING)
    find_package(eckit REQUIRED)
    find_package(atlas REQUIRED)
  else()
    find_package(eckit QUIET)
    if(NOT eckit_FOUND)
      message(STATUS "Could NOT locate eckit.")
    endif()
    find_package(atlas QUIET)
    if(NOT atlas_FOUND)
      message(STATUS "Could NOT locate atlas.")
    endif()
  endif()

  if(atlas_FOUND AND eckit_FOUND)
    message(STATUS "Found atlas and eckit. Setting HAS_UNSTRUCTURED_DEPS=ON.")
    set(HAS_UNSTRUCTURED_DEPS ON)
  else()
    set(STATUS "Cound not find dependencies for unstructured support.")
  endif()

endif()
