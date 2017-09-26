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

include(CheckCXXCompilerFlag)

#.rst:
# dawn_check_cxx_flag
# -------------------
#
# Test if the C++ compiler flag is supported.
#
# .. code-block:: cmake
#
#   dawn_check_cxx_flag(FLAG NAME)
#
# ``FLAG``
#   Compiler flag to check (e.g -O3).
# ``NAME``
#   Name of the check (e.g HAVE_GCC_O3).
#
macro(dawn_check_cxx_flag FLAG NAME)
  check_cxx_compiler_flag("${FLAG}" ${NAME})
endmacro()
