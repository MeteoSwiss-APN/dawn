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

include(FetchContent)

FetchContent_Declare(
    pybind11_fetch
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.4
)
FetchContent_GetProperties(pybind11_fetch)
if(NOT pybind11_fetch_POPULATED)
    message(STATUS "Fetch pybind11 for python-binding")
    FetchContent_Populate(pybind11_fetch)
    add_subdirectory(${pybind11_fetch_SOURCE_DIR} ${pybind11_fetch_BINARY_DIR})
endif()

pybind11_add_module(dawn_python MODULE DawnPython.cpp)
target_link_libraries(dawn_python
  pybind11::pybind11
  DawnASTObjects
  DawnSIRObjects
  DawnSupportObjects
  DawnCompilerObjects
  DawnSerializerObjects
  DawnIIRObjects
  DawnOptimizerObjects
  DawnCodeGenObjects
  ${DAWN_EXTERNAL_LIBRARIES}
)
