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

#.rst:
# dawn_check_in_source_build
# --------------------------
#
# Abort with a meaningful error message if ``CMAKE_BINARY_DIR`` matches ``CMAKE_SOURCE_DIR``.
#
# .. code-block:: cmake
#
#   dawn_check_in_source_build()  
#
function(dawn_check_in_source_build)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
    message(FATAL_ERROR "
  ### In-source builds are not allowed.
  ### Please create a build directory and run cmake from there, 
  ### passing the path to this source directory as the last argument.
  ### This process created the file `CMakeCache.txt' and the 
  ### directory `CMakeFiles'. Please delete them!
")
  endif()
endfunction()