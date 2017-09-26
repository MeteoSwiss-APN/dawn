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

include(DawnIncludeGuard)
dawn_include_guard()

include(CMakeParseArguments)

#.rst:
# dawn_combine_libraries
# ----------------------
#
# Combine multiple object libraries to a single static and, if ``BUILD_SHARED_LIBS`` is ON, shared 
# library. The CMake target of the library is ``<NAME>Static`` and ``<NAME>Shared`` respectively.
# This will also provide an install traget for the libraries.
#
# .. code-block:: cmake
#
#   dawn_combine_libraries(NAME OBJECTS DEPENDS)
#
# ``NAME``
#   Name of the library.
# ``OBJECTS``
#   Object libraries to combine (see .
# ``DEPENDS``
#   List of external libraries and/or CMake targets treated as dependencies of the library.
#
function(dawn_combine_libraries)
  set(options)
  set(one_value_args NAME)
  set(multi_value_args OBJECTS DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  
  if(NOT("${ARG_UNPARSED_ARGUMENTS}" STREQUAL ""))
    message(FATAL_ERROR "dawn_combine_libraries: invalid argument ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT("${ARG_OBJECTS}" STREQUAL ""))
    set(object_sources)
    foreach(object ${ARG_OBJECTS})
      list(APPEND object_sources $<TARGET_OBJECTS:${object}>)
    endforeach()
  endif()

  # Add static library
  add_library(${ARG_NAME}Static STATIC ${object_sources})
  target_link_libraries(${ARG_NAME}Static ${ARG_DEPENDS})
  set_target_properties(${ARG_NAME}Static PROPERTIES OUTPUT_NAME ${ARG_NAME})
  install(TARGETS ${ARG_NAME}Static DESTINATION lib)

  # Add shared library
  if(BUILD_SHARED_LIBS)
    add_library(${ARG_NAME}Shared SHARED ${object_sources})
    target_link_libraries(${ARG_NAME}Shared ${ARG_DEPENDS})
    set_target_properties(${ARG_NAME}Shared PROPERTIES OUTPUT_NAME ${ARG_NAME})
    install(TARGETS ${ARG_NAME}Shared DESTINATION lib)
  endif()
endfunction()
