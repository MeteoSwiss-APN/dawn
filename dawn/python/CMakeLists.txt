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
get_filename_component(camke_dir  "${Protobuf_DIR}" DIRECTORY)
get_filename_component(lib_dir "${camke_dir}" DIRECTORY)
get_filename_component(protobuf_root "${lib_dir}" DIRECTORY)

# Check the python libraries were installed in <protobuf_root>/python/
set(protobuf_python "${protobuf_root}/python")
set(protobuf_init_py "${protobuf_python}/google/protobuf/__init__.py")

if(NOT EXISTS "${protobuf_init_py}")
  message(WARNING
    "Python module of protobuf is not installed correctly! Expected protobuf's __init__.py at:\n"
    "    ${protobuf_init_py}\n"
    "Disabling python module of Dawn."
  )
  set(DAWN_PYTHON OFF CACHE BOOL "" FORCE)
else()
  set(DAWN_INSTALL_PROTOBUF_MODULE "${protobuf_python}")

  set(config_in "${PROJECT_SOURCE_DIR}/python/dawn/config.py.in")
  set(config_out "${PROJECT_BINARY_DIR}/python/dawn/config.py")
  configure_file(${config_in} ${config_out})

  # Copy to build directory
  file(COPY . DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PATTERN "*.py")

  # Copy to install
  install(FILES ${config_out}
          DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/dawn/)
  install(DIRECTORY dawn/
          DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/dawn
          FILES_MATCHING PATTERN "*.py"
          )
endif()

if(DAWN_TESTING)
  if(NOT DEFINED Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter)
  endif()

  add_test(NAME python-test_sir.py COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/dawn/test_sir.py)
endif()
