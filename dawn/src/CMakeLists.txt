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
add_subdirectory(toylib)

if(ENABLE_PYTHON)
  add_subdirectory(dawn4py)
endif()

# install headers
foreach(dir IN ITEMS dawn driver-includes)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inc")
endforeach()
