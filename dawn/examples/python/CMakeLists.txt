
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

function(add_python_example)
  set(options VERIFY)
  set(oneValueArgs NAME)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(example ${ARG_NAME})
  if(ARG_VERIFY)
    set(verify true)
  else()
    set(verify false)
  endif()

  find_program(SH_PROGRAM sh)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python-test.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/python-test-${ARG_NAME}.sh
  )
  add_test(NAME dawn4py-example-${ARG_NAME}.sh
    COMMAND ${CMAKE_COMMAND} -E env
    ${SH_PROGRAM} python-test-${ARG_NAME}.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}
  )
endfunction()

if(${PROJECT_NAME}_TESTING)
  # configure dawn4py in the build directory for testing
  file(COPY ${PROJECT_SOURCE_DIR}/src/dawn4py/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dawn4py)
  add_custom_target(link_dawn4py_examples ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:_dawn4py> ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dawn4py/$<TARGET_FILE_NAME:_dawn4py>
  )

  add_python_example(NAME copy_stencil VERIFY)
  add_python_example(NAME hori_diff_stencil VERIFY)
  add_python_example(NAME tridiagonal_solve_stencil VERIFY)
  add_python_example(NAME tridiagonal_solve_unstructured)
  add_python_example(NAME global_index_stencil)
  add_python_example(NAME unstructured_stencil)
endif()
