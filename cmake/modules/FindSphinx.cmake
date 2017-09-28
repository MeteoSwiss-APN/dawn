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
# FindSphinx
# ----------
#
# .. code-block:: cmake
#
#   find_package(Sphinx [REQUIRED] [QUIET])
#
# This module locates Sphinx_.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# Defines the following variables:
#
# ``SPHINX_FOUND``
#   System has sphinx.
# ``SPHINX_EXECUTABLE``
#   Path to sphinx-build executable.
#
# Hints
# ^^^^^
#
# You can directly set ``SPHINX_EXECUTABLE`` if the module has trouble finding sphinx.
#
# .. _Sphinx: http://www.sphinx-doc.org/en/stable
#

include(FindPackageHandleStandardArgs)

if(NOT DEFINED SPHINX_EXECUTABLE)
  find_program(SPHINX_EXECUTABLE 
    NAMES sphinx-build sphinx-build2 
    DOC "Path to sphinx-build executable"
  )
endif()

find_package_handle_standard_args(sphinx 
  FOUND_VAR 
    SPHINX_FOUND 
  REQUIRED_VARS
    SPHINX_EXECUTABLE
)

mark_as_advanced(SPHINX_EXECUTABLE)