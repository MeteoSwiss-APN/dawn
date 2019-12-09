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

set(config_in ${CMAKE_CURRENT_SOURCE_DIR}/dawn/config.py.in)
set(config_out ${CMAKE_CURRENT_BINARY_DIR}/dawn/config.py)
configure_file(${config_in} ${config_out})

# Copy to build directory
file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILES_MATCHING PATTERN "*.py")

# Copy to install
install(
  DIRECTORY dawn
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/Dawn/python
  FILES_MATCHING PATTERN "*.py"
)
install(
  FILES ${config_out}
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/Dawn/python/dawn
)

if(${PROJECT_NAME}_TESTING)
  if(NOT Python3_FOUND)
    find_package(Python3 COMPONENTS Interpreter)
  endif()

  add_test(NAME python-test_sir.py
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/dawn/test_sir.py)
endif()
