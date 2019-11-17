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

set(config_in "${CMAKE_CURRENT_SOURCE_DIR}/dawn/config.py.in")
set(config_out "${CMAKE_CURRENT_BINARY_DIR}/dawn/config.py")
configure_file(${config_in} ${config_out})

# # Copy to build directory
# file(COPY ${CMAKE_SOURCE_DIR}/python/ DESTINATION ${CMAKE_BINARY_DIR}/python
#       FILES_MATCHING PATTERN "*.py")

# # Copy to install
# install(FILES ${config_out}
#         DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/dawn)
# install(DIRECTORY dawn/
#         DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/dawn
#         FILES_MATCHING PATTERN "*.py"

# )
# endif()
