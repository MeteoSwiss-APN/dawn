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
# FindGSLLibrary
# --------------
#
# Find the The Generic Stencil Language (GSL) libraries and includes.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project::
#
#  GSL_FOUND                - True if GSL was found on the local system.
#  GSL_INCLUDE_DIRS         - Location of the GSL header files.
#  GSL_LIBRARIES            - Location of the (static) GSL libraries (libGSL.a).
#  GSL_HAS_SHARED_LIB       - True if the shared libary of GSL was found.
#  GSL_SHARED_LIBRARIES     - Location of the shared GSL libraries (libGSL.so) if found.
#  GSL_VERSION              - The version of the discovered GSL installation.
#
# Hints
# ^^^^^
#
# Set ``GSL_ROOT`` to a directory that contains a GSL installation.
#
# This script expects to find libraries at ``$GSL_ROOT/lib`` and the GSL headers at 
# ``$GSL_ROOT/include``.
#

include(FindPackageHandleStandardArgs)

#
# Parse GSL_ROOT
#
set(GSL_ROOT_ENV $ENV{GSL_ROOT})
if(GSL_ROOT_ENV)
  set(GSL_ROOT ${GSL_ROOT_ENV} CACHE PATH "GSL install path.")
endif()

if(NOT(DEFINED GSL_ROOT))
  find_path(GSL_ROOT NAMES include/gsl/Support/Config.h)
else()
  get_filename_component(_GSL_ROOT_ABSOLUTE ${GSL_ROOT} ABSOLUTE)
  set(GSL_ROOT ${_GSL_ROOT_ABSOLUTE} CACHE PATH "GSL install path.")
endif()

#
# Default version is 0.0.0
#
if(NOT GSLLibrary_FIND_VERSION)
  if(NOT GSLLibrary_FIND_VERSION_MAJOR)
    set(GSLLibrary_FIND_VERSION_MAJOR 0)
  endif()
  if(NOT GSLLibrary_FIND_VERSION_MINOR)
    set(GSLLibrary_FIND_VERSION_MINOR 0)
  endif()
  if(NOT GSLLibrary_FIND_VERSION_PATCH)
    set(GSLLibrary_FIND_VERSION_PATCH 0)
  endif()
  set(GSLLibrary_FIND_VERSION 
      "${GSLLibrary_FIND_VERSION_MAJOR}.${GSLLibrary_FIND_VERSION_MINOR}.${GSLLibrary_FIND_VERSION_PATCH}")
endif()

#===---------------------------------------------------------------------------------------------===
#   Find GSL headers
#====--------------------------------------------------------------------------------------------===
if(GSL_ROOT)
  find_path(GSL_INCLUDE_DIRS NAMES gsl/Support/Config.h HINTS ${GSL_ROOT}/include)
endif()

#===---------------------------------------------------------------------------------------------===
#   Read config file (gsl/Support/Config.h)
#====--------------------------------------------------------------------------------------------===
if(GSL_ROOT)
  file(READ ${GSL_ROOT}/include/gsl/Support/Config.h _CONFIG_FILE)

  # Get version  
  string(REGEX MATCH "define[ \t]+GSL_VERSION_MAJOR[ \t]+([0-9]+)" _MAJOR "${_CONFIG_FILE}")
  set(GSL_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+GSL_VERSION_MINOR[ \t]+([0-9]+)" _MINOR "${_CONFIG_FILE}")
  set(GSL_MINOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+GSL_VERSION_PATCH[ \t]+([0-9]+)" _PATCH "${_CONFIG_FILE}")
  set(GSL_PATCH_VERSION "${CMAKE_MATCH_1}")

  set(GSL_VERSION "${GSL_MAJOR_VERSION}.${GSL_MINOR_VERSION}.${GSL_PATCH_VERSION}")
endif()

#===---------------------------------------------------------------------------------------------===
#   Find GSL libraries
#====--------------------------------------------------------------------------------------------===
if(GSL_ROOT)
    find_library(GSL_LIBRARIES NAMES "libGSL.a" HINTS ${GSL_ROOT}/lib)
    find_library(shared_lib NAMES "libGSL${CMAKE_SHARED_LIBRARY_SUFFIX}" HINTS ${GSL_ROOT}/lib)
    mark_as_advanced(shared_lib)
    if(shared_lib)
      set(GSL_HAS_SHARED_LIB TRUE)
      set(GSL_SHARED_LIBRARIES ${shared_lib})
    else()
      set(GSL_HAS_SHARED_LIB FALSE)
      set(GSL_SHARED_LIBRARIES "")
    endif()
endif()

#===---------------------------------------------------------------------------------------------===
# Report result 
#====--------------------------------------------------------------------------------------------===
find_package_handle_standard_args(
  GSL
  FAIL_MESSAGE "Could NOT find GSL. (Try setting GSL_ROOT in the env)"
  REQUIRED_VARS  GSL_ROOT
                 GSL_INCLUDE_DIRS
                 GSL_LIBRARIES
                 GSL_HAS_SHARED_LIB
                 GSL_SHARED_LIBRARIES
  VERSION_VAR GSL_VERSION)

mark_as_advanced(GSL_INCLUDE_DIRS
                 GSL_LIBRARIES
                 GSL_HAS_SHARED_LIB
                 GSL_SHARED_LIBRARIES
                 GSL_VERSION)

if(GSL_FOUND)
  message(STATUS "GSL version: ${GSL_VERSION}")
else()
  if(${GSLLibrary_FIND_REQUIRED})
     message(FATAL_ERROR "Could NOT find GSL. (Try setting GSL_ROOT in the env)")
  endif()
endif()

