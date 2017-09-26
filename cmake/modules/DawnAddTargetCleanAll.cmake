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

include(DawnGetScriptDir)

#.rst:
# dawn_add_target_clean_all
# -------------------------
#
# Provide a ``clean-all`` target which clears the CMake cache and all related CMake files and 
# directories. This effectively removes the following files/directories:
#
#    - ``${CMAKE_BINARY_DIR}/CMakeCache.txt``
#    - ``${CMAKE_BINARY_DIR}/CTestTestfile.cmake``
#    - ``${CMAKE_BINARY_DIR}/cmake_install.cmake``
#    - ``${CMAKE_BINARY_DIR}/CMakeFiles``
#
# .. code-block:: cmake
#
#  dawn_add_target_clean_all([ARGN...])
#
# ``ARGN``
#   Addtional files or directories to remove.
#
function(dawn_add_target_clean_all)
  dawn_get_script_dir(script_dir)
  set(dawn_add_target_clean_all_extra_args ${ARGN})

  set(input_script ${script_dir}/DawnAddTargetCleanAll-Script.cmake.in)
  set(output_script ${CMAKE_BINARY_DIR}/dawn-cmake/cmake/DawnAddTargetCleanAll-Script.cmake)

  # Configure the script
  configure_file(${input_script} ${output_script} @ONLY)

  add_custom_target(clean-all
      COMMAND ${CMAKE_MAKE_PROGRAM} clean
      COMMAND ${CMAKE_COMMAND} -P "${output_script}"
      COMMENT "Removing CMake related files"
  )
endfunction()