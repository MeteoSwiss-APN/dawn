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

# Define SIR and IIR proto files here, since they are needed for the python bindings in dawn4py as well as dawn
# This situation will be fixed when dawn_irs is separated from dawn
set(dawn_sir_proto_files SIR/SIR.proto SIR/statements.proto SIR/enums.proto)
set(dawn_iir_proto_files IIR/IIR.proto)

add_subdirectory(dawn)
add_subdirectory(dawn-c)

if(ENABLE_PYTHON)
  add_subdirectory(dawn4py)
endif()

# install headers
install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inc" PATTERN "dawn4py" EXCLUDE)
