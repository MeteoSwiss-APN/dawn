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
# .. _dawn_add_library:
#
# dawn_add_library
# ----------------
#
# Compile the given sources into an archive or object library. This will provide the CMake targets
# ``<NAME>Objects``, ``<NAME>Static`` and ``<NAME>Shared`` respectively. 
#
# .. code-block:: cmake
#
#   dawn_add_library(NAME SOURCES DEPENDS [OUTPUT_DIR])
#
# ``NAME``
#   Name of the exectuable as well as the CMake target to build it.
# ``SOURCES``
#   List of source files making up the exectuable.
# ``DEPENDS``
#   List of external libraries and/or CMake targets to link against.
# ``OBJECT`` [optional]
#   Create a CMake object library.
# ``ARCHIVE`` [optional] 
#   Create a static and, if ``BUILD_SHARED_LIBS`` is ON, shared archive.
#
function(dawn_add_library)  
  set(options OBJECT ARCHIVE)
  set(one_value_args NAME)
  set(multi_value_args SOURCES DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  
  if(NOT("${ARG_UNPARSED_ARGUMENTS}" STREQUAL ""))
    message(FATAL_ERROR "invalid argument ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  
  if(ARG_OBJECT)
   add_library(${ARG_NAME}Objects OBJECT ${ARG_SOURCES})
  endif()
  
  if(ARG_ARCHIVE)
    add_library(${ARG_NAME}Static STATIC ${ARG_SOURCES})
    target_link_libraries(${ARG_NAME}Static ${ARG_DEPENDS})
    set_target_properties(${ARG_NAME}Static PROPERTIES OUTPUT_NAME ${ARG_NAME})

    if(BUILD_SHARED_LIBS)
      add_library(${ARG_NAME}Shared SHARED ${ARG_SOURCES})
      target_link_libraries(${ARG_NAME}Shared ${ARG_DEPENDS})
      set_target_properties(${ARG_NAME}Shared PROPERTIES OUTPUT_NAME ${ARG_NAME})
    endif()
  endif()
endfunction()