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

#.rst:
# Findclang-format
# ----------------
#
# .. code-block:: cmake
#
#   find_package(clang-format [REQUIRED] [QUIET])
#
# This module locates clang-format_. ``CLANG-FORMAT_FOUND`` will report if clang-format has been 
# found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# Defines the following variables:
#
# ``CLANG-FORMAT_FOUND``
#   System has clang-format.
# ``CLANG-FORMAT_EXECUTABLE``
#   Path to the clang-format executable.
# ``CLANG-FORMAT_VERSION``
#   Version of clang-format.
#
# Hints
# ^^^^^
#
# You can directly set ``CLANG-FORMAT_EXECUTABLE`` if the module has trouble finding clang-format.
#
# .. _clang-format: https://clang.llvm.org/docs/ClangFormat.html
#
include(FindPackageHandleStandardArgs)

# Find clang-format
if(NOT DEFINED CLANG-FORMAT_EXECUTABLE)
  find_program(CLANG-FORMAT_EXECUTABLE 
    NAMES clang-format-${clang-format_FIND_VERSION}
          clang-format
    DOC "Path to clang-format executable"
  )
endif()
  
# Extract version
if(CLANG-FORMAT_EXECUTABLE)
  execute_process(
    COMMAND ${CLANG-FORMAT_EXECUTABLE} --version
    RESULT_VARIABLE returncode
    OUTPUT_VARIABLE stdout
  )

  if("${returncode}" STREQUAL "0")
    string(REGEX MATCH "^clang-format version ([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*)$" match "${stdout}")
    set(major "${CMAKE_MATCH_1}") 
    set(minor "${CMAKE_MATCH_2}") 
    set(CLANG-FORMAT_VERSION "${major}.${minor}")
  endif()
endif()

find_package_handle_standard_args(clang-format 
  FOUND_VAR 
    CLANG-FORMAT_FOUND 
  REQUIRED_VARS 
    CLANG-FORMAT_EXECUTABLE
  VERSION_VAR
    CLANG-FORMAT_VERSION
)
