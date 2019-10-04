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
# Findbash
# --------
#
# .. code-block:: cmake
#
#   find_package(bash [REQUIRED] [QUIET])
#
# This module locates bash_. ``BASH_FOUND`` will report if bash has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# Defines the following variables:
#
# ``BASH_FOUND``
#   System has bash.
# ``BASH_EXECUTABLE``
#   Path to the bash executable.
#
# Hints
# ^^^^^
#
# You can directly set ``BASH_EXECUTABLE`` if the module has trouble finding bash.
#
# .. _bash: https://www.gnu.org/software/bash/
#
include(FindPackageHandleStandardArgs)

if(NOT DEFINED BASH_EXECUTABLE)
  find_program(BASH_EXECUTABLE 
    NAMES bash
    DOC "Path to bash executable"
  )
endif()

find_package_handle_standard_args(bash 
  FOUND_VAR 
    BASH_FOUND 
  REQUIRED_VARS 
    BASH_EXECUTABLE
)

mark_as_advanced(BASH_EXECUTABLE)

