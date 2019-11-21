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

set(DAWN_INSTALL_MODULE ${CMAKE_INSTALL_PREFIX}/python)
set(DAWN_INSTALL_DAWNCLIB ${CMAKE_INSTALL_FULL_LIBDIR}/$<TARGET_FILE_NAME:DawnC>)

# This two-step hack is necessary because there are both variables and generator
# expressions necessary to generate config.py
configure_file(${config_in} ${CMAKE_CURRENT_BINARY_DIR}/.tmp)
file(GENERATE OUTPUT ${config_out} INPUT ${CMAKE_CURRENT_BINARY_DIR}/.tmp)

# Copy to install
install(
  FILES ${config_out}
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/Dawn/examples/python
)
install(
  DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/Dawn/examples/python
)
