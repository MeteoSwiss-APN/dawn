##===------------------------------------------------------------------------------*- CMake -*-===##
##                          _                      
##                         | |                     
##                       __| | __ ___      ___ __  
##                      / _` |/ _` \ \ /\ / / '_ \ 
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

#.rst:
# dawn_cmake_init
# ---------------
#
# Add the necessary paths to ``CMAKE_MODULE_PATH`` to use the functions, macros and modules of the 
# ``dawn`` project. To find the ``dawn`` modules the following directories are searched:
#
#  1. CMake variable ``DAWN_ROOT``
#  2. Environment variable ``DAWN_ROOT``
#  3. ``${CMAKE_CURRENT_LIST_DIR}/../dawn``
#
# where ``CMAKE_CURRENT_LIST_DIR`` is the directory of the listfile currently being processed. Note 
# that this script should be copied into each sub-project. 
#
# .. code-block:: cmake
#
#  include(DawnCMakeInit)
#  dawn_cmake_init()
#
macro(dawn_cmake_init)
  set(dawn_dir)
  
  if(DEFINED DAWN_ROOT)
    set(dawn_dir ${DAWN_ROOT})
  elseif(NOT "$ENV{DAWN_ROOT}" STREQUAL "")
    set(dawn_dir $ENV{DAWN_ROOT})
  elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../dawn")
    set(dawn_dir ${CMAKE_CURRENT_LIST_DIR}/../dawn)
    message(FATAL_ERROR "Could NOT find Dawn. (Try setting DAWN_ROOT in the env)")
  endif()

  get_filename_component(dawn_dir ${dawn_dir} ABSOLUTE)
  list(APPEND CMAKE_MODULE_PATH "${dawn_dir}/cmake")
endmacro()