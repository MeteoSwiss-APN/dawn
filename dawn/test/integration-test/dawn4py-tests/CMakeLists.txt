
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

find_program(SH_PROGRAM sh)

function(add_python_example)
  set(options VERIFY)
  set(oneValueArgs NAME)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_test(NAME Dawn4Py::Integration::${ARG_NAME}
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${DAWN4PY_MODULE_DIR}:${PROTOBUF_PYTHON_DIR}
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_NAME}.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}
  )

  if(ARG_VERIFY)
    add_test(NAME Dawn4Py::Integration::${ARG_NAME}-verify
      COMMAND ${CMAKE_COMMAND} -E compare_files ${ARG_NAME}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/data/${ARG_NAME}_ref.cpp
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}
    )
    set_tests_properties(Dawn4Py::Integration::${ARG_NAME}-verify PROPERTIES DEPENDS Dawn4Py::Integration::${ARG_NAME})
  endif()
endfunction()

add_python_example(NAME copy_stencil VERIFY)
add_python_example(NAME hori_diff_stencil VERIFY)
#add_python_example(NAME tridiagonal_solve_stencil VERIFY)
add_python_example(NAME ICON_laplacian_stencil VERIFY)
add_python_example(NAME tridiagonal_solve_unstructured)
add_python_example(NAME global_index_stencil)
add_python_example(NAME unstructured_stencil)
