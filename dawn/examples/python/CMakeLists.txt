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

set(config_in "${CMAKE_CURRENT_SOURCE_DIR}/config.py.in")
set(config_out "${CMAKE_CURRENT_BINARY_DIR}/config.py")

# This two-step hack is necessary because there are both variables and generator
# expressions necessary to generate config.py
set(DAWN_INSTALL_MODULE ${CMAKE_INSTALL_PREFIX}/python)
set(DAWN_INSTALL_DAWNCLIB ${CMAKE_INSTALL_FULL_LIBDIR}/$<TARGET_FILE_NAME:DawnC>)
configure_file(${config_in} ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config.py.tmp @ONLY)
file(GENERATE OUTPUT ${config_out} INPUT ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config.py.tmp)

# Copy to install
install(
  FILES ${config_out}
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/Dawn/examples/python
)
install(
  DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/Dawn/examples/python
)

if(${PROJECT_NAME}_TESTING)
  # configure python examples in the build directory for testing
  set(examples_build_dir "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/for_test")
  set(config_out "${examples_build_dir}/config.py")
  set(DAWN_INSTALL_MODULE ${PROJECT_BINARY_DIR}/python/)
  set(DAWN_INSTALL_DAWNCLIB $<TARGET_FILE:DawnC>)
  configure_file(${config_in} ${examples_build_dir}/config.py.tmp @ONLY)
  file(GENERATE OUTPUT ${config_out} INPUT ${examples_build_dir}/config.py.tmp)
  file(COPY . DESTINATION ${examples_build_dir})

  set(examples copy_stencil hori_diff tridiagonal_solve unstructured_stencil)
  foreach(example ${examples})
    add_test(NAME "python-test-${example}.py"
      COMMAND
        ${CMAKE_COMMAND}
        -DEXAMPLE=${example}
        -DDAWN_PYTHON_MODULES_DIR=${DAWN_INSTALL_MODULE}
        -DDAWN_PYTHON_EXAMPLES_DIR=${examples_build_dir}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_python.cmake)
  endforeach()
endif()
