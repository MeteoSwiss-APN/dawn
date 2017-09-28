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

set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})
set(SPHINX_ROOT ${CMAKE_CURRENT_LIST_DIR})

# Extract CMake documentation
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules-generate.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/cmake-modules-generate.cmake @ONLY)
add_custom_target(DawnDocsCMakeModules
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake-modules-generate.cmake
  COMMENT "Configuring dawn-cmake" VERBATIM
)

# Sphinx
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
               ${CMAKE_CURRENT_SOURCE_DIR}/conf.py @ONLY)
add_custom_target(DawnDocsSphinx ALL
   COMMAND ${SPHINX_EXECUTABLE} -b html -d ${CMAKE_BINARY_DIR}/sphinx/doctrees 
           ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/html
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
   COMMENT "Generating sphinx documentation" VERBATIM
   DEPENDS 
    conf.py
    index.rst
)

add_dependencies(DawnDocsSphinx DawnDocsCMakeModules)

# Docs target
add_custom_target(docs ALL
 COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/sphinx/html 
                                            ${CMAKE_SOURCE_DIR}/docs/html
 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
 COMMENT "Generating documentation" VERBATIM
 DEPENDS DawnDocsSphinx
)