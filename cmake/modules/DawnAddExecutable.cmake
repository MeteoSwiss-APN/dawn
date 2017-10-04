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
# dawn_add_executable
# -------------------
#
# Compile the given objects into a runnable executable (.exe).
#
# .. code-block:: cmake
#
#   dawn_add_executable(NAME SOURCES DEPENDS [OUTPUT_DIR])
#
# ``NAME``
#   Name of the exectuable as well as the CMake target to build it.
# ``SOURCES``
#   List of source files making up the exectuable.
# ``DEPENDS``
#   List of external libraries and/or CMake targets to link against.
# ``OUTPUT_DIR`` [optional]
#   Directory to place the exectuable (e.g ``${CMAKE_BINARY_DIR}/bin``).
# ``INSTALL_DESTINATION`` [optional]
#   Destition (relative to ``CMAKE_INSTALL_PREFIX``) to install the executable.
#
function(dawn_add_executable)
  set(one_value_args NAME OUTPUT_DIR INSTALL_DESTINATION)
  set(multi_value_args SOURCES DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT("${ARG_UNPARSED_ARGUMENTS}" STREQUAL ""))
    message(FATAL_ERROR "dawn_add_executable: invalid argument ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  
  add_executable(${ARG_NAME} ${ARG_SOURCES})
  target_link_libraries(${ARG_NAME} ${ARG_DEPENDS})

  if(ARG_OUTPUT_DIR)
    set_target_properties(${ARG_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIR}")
  endif()

  if(ARG_INSTALL_DESTINATION)
    install(TARGETS ${ARG_NAME} 
            DESTINATION ${ARG_INSTALL_DESTINATION} 
            EXPORT ${ARG_NAME}Targets)
  endif()
endfunction()
