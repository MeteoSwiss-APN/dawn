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
# dawn_configure_file
# -------------------
#
# Configure a file and store the output in the same directory hirarchy as the input file while 
# substituting ``CMAKE_SOURCE_DIR`` for ``CMAKE_BINARY_DIR``. If the input file has a ``.cmake`` 
# extension, it will be removed in the output. This will also handle the installation of the file 
# correctly by replacing ``src/`` with ``include/``.
#
# .. code-block:: cmake
#
#   dawn_configure_file(FILE)
#
# ``FILE``
#   File to configure.
#
# Example
# ^^^^^^^
# 
# .. code-block:: cmake
#
#   dawn_configure_file(${CMAKE_SOURCE_DIR}/src/dawn/Config.h.cmake)
#
# Ths stores the configured file in ``${CMAKE_BINARY_DIR}/src/dawn/Config.h`` and, in addition, it 
# will install it in ``${CMAKE_INSTALL_PREFIX}/include/dawn/Config.h``" (hence it will strip 
# ``"src/"``).
#
function(dawn_configure_file FILE)
  get_filename_component(absolute_path ${FILE} ABSOLUTE)
  file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ${absolute_path})

  get_filename_component(output_dir ${relative_path} DIRECTORY)
  get_filename_component(output_name_cmake ${relative_path} NAME)
  string(REGEX REPLACE ".cmake" "" output_file_name ${output_name_cmake})
  
  set(input_file ${FILE})
  set(output_file "${CMAKE_BINARY_DIR}/${output_dir}/${output_file_name}")
  
  configure_file(${input_file} ${output_file})
  
  # Install file
  string(REGEX MATCH "src/(.*)" unused ${output_dir})
  set(install_dir ${CMAKE_MATCH_1}) 
  install(FILES ${output_file} DESTINATION include/${install_dir})
endfunction()
