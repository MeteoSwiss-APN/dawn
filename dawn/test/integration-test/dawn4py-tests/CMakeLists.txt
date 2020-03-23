
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

#TODO: move to unit tests (and change ::Integration:: below)
#TODO: move *_ref.cpp to CodeGen test refs

function(add_dawn4py_test)
  set(oneValueArgs NAME)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_test(NAME Dawn4Py::Integration::${ARG_NAME}
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${DAWN4PY_MODULE_DIR}:${PROTOBUF_PYTHON_DIR}
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_NAME}.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME Dawn4Py::Integration::${ARG_NAME}-verify
    COMMAND ${CMAKE_COMMAND} -E compare_files ${ARG_NAME}.sir ${CMAKE_CURRENT_SOURCE_DIR}/data/${ARG_NAME}_ref.sir
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(Dawn4Py::Integration::${ARG_NAME}-verify PROPERTIES DEPENDS Dawn4Py::Integration::${ARG_NAME})

endfunction()

add_dawn4py_test(NAME copy_stencil)
add_dawn4py_test(NAME hori_diff_stencil)
add_dawn4py_test(NAME tridiagonal_solve_stencil)
add_dawn4py_test(NAME ICON_laplacian_stencil)
add_dawn4py_test(NAME tridiagonal_solve_unstructured)
add_dawn4py_test(NAME global_index_stencil)
add_dawn4py_test(NAME simple_reduction_stencil)
add_dawn4py_test(NAME sparse_dimensions)
