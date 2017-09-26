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
# Findccache
# ----------
#
# .. code-block:: cmake
#
#   find_package(ccache [REQUIRED] [QUIET])
#
# This module locates ccache_. ``CCACHE_FOUND`` will report if ccache has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# Defines the following variables:
#
# ``CCACHE_FOUND``
#   System has ccache.
# ``CCACHE_EXECUTABLE``
#   Path to the ccache executable.
#
# Hints
# ^^^^^
#
# You can directly set ``CCACHE_EXECUTABLE`` if the module has trouble finding ccache.
#
# .. _ccache: https://ccache.samba.org

include(FindPackageHandleStandardArgs)

if(NOT DEFINED CCACHE_EXECUTABLE)
  find_program(CCACHE_EXECUTABLE 
    NAMES ccache
    DOC "Path to ccache executable"
  )
endif()

find_package_handle_standard_args(ccache 
  FOUND_VAR 
    CCACHE_FOUND 
  REQUIRED_VARS 
    CCACHE_EXECUTABLE
)

mark_as_advanced(CCACHE_EXECUTABLE)

