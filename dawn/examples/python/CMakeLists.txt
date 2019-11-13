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

# Protobuf_DIR is <protobuf_root>/lib/cmake/protobuf
get_filename_component(cmake_dir  "${Protobuf_DIR}" DIRECTORY)
get_filename_component(lib_dir "${cmake_dir}" DIRECTORY)
get_filename_component(protobuf_root "${lib_dir}" DIRECTORY)

#
# Check the python libraries were installed in <protobuf_root>/python/
set(protobuf_python "${protobuf_root}/python")
set(protobuf_init_py "${protobuf_python}/google/protobuf/__init__.py")

if(NOT EXISTS "${protobuf_init_py}")
  message(WARNING
    "Python module of protobuf is not installed correctly! Expected protobuf's __init__.py at:\n"
    "    ${protobuf_init_py}\n"
    "Disabling python module of Dawn."
  )
else()
  set(DAWN_INSTALL_PROTOBUF_MODULE "${protobuf_python}")

  # install python examples
  set(config_in "${CMAKE_SOURCE_DIR}/examples/python/config.py.in")
  set(config_out "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/for_install/config.py")
  set(DAWN_INSTALL_MODULE ${CMAKE_INSTALL_PREFIX}/python/)
  set(DAWN_INSTALL_DAWNCLIB ${CMAKE_INSTALL_PREFIX}/lib/libDawnC.so)
  configure_file(${config_in} ${config_out})
  install(FILES ${config_out}
	  DESTINATION ${DAWN_INSTALL_EXAMPLES_DIR}/python)
  install(DIRECTORY .
	  DESTINATION ${DAWN_INSTALL_EXAMPLES_DIR}/python/
  )

  if(DAWN_TESTING)
    # configure python examples in the build directory for testing
    set(examples_build_dir "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/for_test")
    set(config_in "${CMAKE_SOURCE_DIR}/examples/python/config.py.in")
    set(config_out "${examples_build_dir}/config.py")
    set(DAWN_INSTALL_MODULE ${PROJECT_BINARY_DIR}/python/)
    set(DAWN_INSTALL_DAWNCLIB ${PROJECT_BINARY_DIR}/src/dawn-c/libDawnC.so)
    configure_file(${config_in} ${config_out})
    file(COPY . DESTINATION ${examples_build_dir})

    if(NOT DEFINED PYTHON_EXECUTABLE)
      find_package(PythonInterp 3 REQUIRED)
    endif()

    set(examples copy_stencil hori_diff tridiagonal_solve unstructured_stencil)
    set(verified_examples copy_stencil hori_diff tridiagonal_solve)
    foreach(example ${examples})
      if(${example} IN_LIST verified_examples)
        set(verify ON)
      else()
        set(verify OFF)
      endif()
      add_test(NAME "python-test-${example}.py"
        COMMAND
          cmake
          -DEXAMPLE=${example}
          -DVERIFY=${verify}
          -DDAWN_PYTHON_MODULES_DIR=${DAWN_INSTALL_MODULE}
          -DDAWN_PYTHON_EXAMPLES_DIR=${examples_build_dir}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/test_python.cmake)
    endforeach()
  endif()
endif()
