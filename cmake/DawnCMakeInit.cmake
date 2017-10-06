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
# dawn_cmake_init
# ---------------
#
# Add the necessary paths to ``CMAKE_MODULE_PATH`` to use the functions, macros and modules of the 
# ``Dawn`` project. To find the ``Dawn`` modules the following directories are searched:
#
#  1. Check for ``DAWN_ROOT`` (pointing to the root folder of the installation)
#  2. Check for ``Dawn_DIR`` (pointing to the directory containing ``DawnConfig.cmake``, equivalent 
#     to ``${CMAKE_ROOT}/cmake``).
#  3. Check ``${CMAKE_CURRENT_LIST_DIR}/../dawn``
#
# where ``CMAKE_CURRENT_LIST_DIR`` is the directory of the listfile currently being processed. Note 
# that this only checks for the CMake directory in thus also usuable with the source directory Dawn. 
#
# .. code-block:: cmake
#
#  include(DawnCMakeInit)
#  dawn_cmake_init()
#
macro(dawn_cmake_init)
  set(dawn_cmake_dir)
  
  # If Dawn_DIR points to the root directory (instead of <dawn-dir>/cmake), we correct this here
  get_filename_component(dawn_config_file "${Dawn_DIR}/cmake/DawnConfig.cmake" ABSOLUTE)
  if(DEFINED Dawn_DIR AND EXISTS "${dawn_config_file}")
    set(Dawn_DIR "${Dawn_DIR}/cmake" CACHE PATH "Path to DawnConfig.cmake" FORCE)
  endif()  

  if(DEFINED DAWN_ROOT)
    set(dawn_cmake_dir "${DAWN_ROOT}/cmake/modules")
  elseif(NOT "$ENV{DAWN_ROOT}" STREQUAL "")
    set(dawn_cmake_dir "$ENV{DAWN_ROOT}/cmake/modules")
  elseif(DEFINED Dawn_DIR)
    set(dawn_cmake_dir "${Dawn_DIR}/modules")
  elseif(NOT "$ENV{Dawn_DIR}" STREQUAL "")
    set(dawn_cmake_dir "$ENV{Dawn_DIR}/modules")
  elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../dawn")
    set(dawn_cmake_dir "${CMAKE_CURRENT_LIST_DIR}/../dawn/cmake/modules")
    message(FATAL_ERROR "Could NOT find Dawn. (Try setting DAWN_ROOT in the env)")
  endif()

  get_filename_component(dawn_cmake_dir ${dawn_cmake_dir} ABSOLUTE)
  
  # Sanity check the CMake directory
  if(NOT EXISTS ${dawn_cmake_dir})
    message(FATAL_ERROR "Invalid Dawn directory: ${dawn_cmake_dir} (missing dawn/cmake/modules)")
  endif()

  list(APPEND CMAKE_MODULE_PATH "${dawn_cmake_dir}")
endmacro()
