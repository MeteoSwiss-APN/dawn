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

yoda_combine_libraries(
  NAME dawn_python
  OBJECTS DawnSupportObjects
          DawnCompilerObjects
          DawnIIRObjects
          DawnOptimizerObjects
          DawnCodeGenObjects
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  VERSION ${DAWN_VERSION}
  DEPENDS ${DAWN_EXTERNAL_LIBRARIES} DawnSIRStatic DawnSerializerStatic pybind11::pybind11
  EXPORT_GROUP DawnTargets
)
set_target_properties(dawn_pythonStatic PROPERTIES PREFIX "")
set_target_properties(dawn_pythonShared PROPERTIES PREFIX "")
install(TARGETS pybind11 DESTINATION ${DAWN_INSTALL_LIB_DIR} EXPORT DawnTargets)
