##===------------------------------------------------------------------------------*- CMake -*-===##
##                         _     _ _              _            _                   
##                        (_)   | | |            | |          | |                  
##               __ _ _ __ _  __| | |_ ___   ___ | |___    ___| | __ _ _ __   __ _ 
##              / _` | '__| |/ _` | __/ _ \ / _ \| / __|  / __| |/ _` | '_ \ / _` |
##             | (_| | |  | | (_| | || (_) | (_) | \__ \ | (__| | (_| | | | | (_| |
##              \__, |_|  |_|\__,_|\__\___/ \___/|_|___/  \___|_|\__,_|_| |_|\__, |
##               __/ |                                                        __/ |
##              |___/                                                        |___/ 
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

#.rst:
# FindGridTools
# -------------
#
# This script locates gridtools. This script makes use of the standard find_package arguments of
# ``VERSION``, ``REQUIRED`` and ``QUIET``. GRIDTOOLS_FOUND will report if an acceptable version of
# gridtools was found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# Defines the following variables:
#
#   GRIDTOOLS_FOUND           - System has GridTools libraries and headers
#   GRIDTOOLS_INCLUDE_DIRS    - The location of GridTools headers
#
#
# Hints
# ^^^^^
#
# Set ``GRIDTOOLS_ROOT`` to a directory that contains a gridtools installation, or directly set 
# ``GRIDTOOLS_INCLUDE_DIRS``.

include(FindPackageHandleStandardArgs)

#
# Parse GRIDTOOLS_ROOT
#
set(GRIDTOOLS_ROOT_ENV $ENV{GRIDTOOLS_ROOT})
if(GRIDTOOLS_ROOT_ENV)
  set(GRIDTOOLS_ROOT ${GRIDTOOLS_ROOT_ENV} CACHE PATH "gridtools install path.")
endif()

if(NOT(GRIDTOOLS_ROOT))
  find_path(GRIDTOOLS_ROOT NAMES include/gridtools.hpp HINTS ${GRIDTOOLS_INCLUDE_DIRS})
else()
  get_filename_component(_GRIDTOOLS_ROOT_ABSOLUTE ${GRIDTOOLS_ROOT} ABSOLUTE)
  set(GRIDTOOLS_ROOT ${_GRIDTOOLS_ROOT_ABSOLUTE} CACHE PATH "gridtools install path.")
endif()

#===---------------------------------------------------------------------------------------------===
#   Find gridtools headers
#====--------------------------------------------------------------------------------------------===
if(NOT(GRIDTOOLS_INCLUDE_DIRS))
  find_path(GRIDTOOLS_INCLUDE_DIRS NAMES gridtools.hpp HINTS ${GRIDTOOLS_ROOT}/include)
endif()

#===---------------------------------------------------------------------------------------------===
#   Report result 
#====--------------------------------------------------------------------------------------------===
find_package_handle_standard_args(
  gridtools
  FAIL_MESSAGE "Could NOT find gridtools. (Try setting GRIDTOOLS_ROOT in the env)"
  REQUIRED_VARS  GRIDTOOLS_INCLUDE_DIRS
)

mark_as_advanced(GRIDTOOLS_INCLUDE_DIRS)

if(NOT(GRIDTOOLS_FOUND) AND GridTools_FIND_REQUIRED)
  message(FATAL_ERROR "Could NOT find gridtools. (Try setting GRIDTOOLS_ROOT in the env)")
endif()

