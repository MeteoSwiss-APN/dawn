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

set(config_in ${CMAKE_CURRENT_SOURCE_DIR}/config.py.in)
set(config_out ${CMAKE_CURRENT_BINARY_DIR}/config.py)

set(DAWN_BUILD_MODULE ${PROJECT_BINARY_DIR}/python)
set(DAWN_BUILD_DAWNCLIB $<TARGET_FILE:DawnCShared>)
configure_file(${config_in} ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config.py.tmp @ONLY)
file(GENERATE OUTPUT ${config_out} INPUT ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config.py.tmp)

# Copy to install
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/Dawn/examples/python
)
install(
  FILES ${config_out}
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/Dawn/examples/python
)

if(${PROJECT_NAME}_TESTING)
  # configure python examples in the build directory for testing
  set(examples_build_dir ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY})

  find_program (SH_PROGRAM sh)

  # TODO unstructured_stencil
  set(examples
    copy_stencil
    hori_diff
    tridiagonal_solve
  )
  foreach(example ${examples})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python-test.sh.in
      ${examples_build_dir}/python-test-${example}.sh
    )
    add_test(NAME python-test-${example}.sh
      COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${PROJECT_BINARY_DIR}/python:${CMAKE_CURRENT_BINARY_DIR}
      ${SH_PROGRAM} python-test-${example}.sh
      WORKING_DIRECTORY ${examples_build_dir}
    )
  endforeach()
endif()
