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

add_subdirectory(dawn)
add_subdirectory(dawn-c)

if(DAWN_PYTHON)
  add_subdirectory(dawn4py)
endif()

# Add a dummy target with the interface headers for correct IDE parsing
yoda_add_custom_dummy_target(NAME DawnParseInterfaceHeaders
                             DIRECTORIES "interface")

# Add a dummy target with the driver-includes headers for correct IDE parsing
yoda_add_custom_dummy_target(NAME DawnParseDriverIncludes
                          DIRECTORIES "driver-includes")
