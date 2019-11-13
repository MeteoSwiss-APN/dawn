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

pybind11_add_module(_dawn4py MODULE _dawn4py.cpp)
target_link_libraries(_dawn4py PUBLIC
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
target_compile_options(_dawn4py PUBLIC -Wno-shadow)

# Generate python proto files
# set(dawn_iir_proto "${CMAKE_SOURCE_DIR}/src/dawn/IIR/proto")
# dawn_protobuf_generate(
#   OUT_FILES iir_proto_python_files
#   PROTOS IIR/IIR.proto
#   WDIR ${dawn_iir_proto}
#   PACKG SIR
#   LANGUAGE python
# )
# add_custom_target(Dawn4PyIIRProto ALL
#   COMMAND ${CMAKE_COMMAND} -E copy ${iir_proto_python_files} ${CMAKE_CURRENT_SOURCE_DIR}/serialization/IIR
#   DEPENDS ${iir_proto_python_files}
# )

# set(dawn_sir_proto "${CMAKE_SOURCE_DIR}/src/dawn/SIR/proto")
# dawn_protobuf_generate(
#   OUT_FILES sir_proto_python_files
#   PROTOS SIR/SIR.proto SIR/statements.proto
#   WDIR ${dawn_sir_proto}
#   PACKG SIR
#   LANGUAGE python
# )
# add_custom_target(Dawn4PySIRProto ALL
#   COMMAND ${CMAKE_COMMAND} -E copy ${sir_proto_python_files} ${CMAKE_CURRENT_SOURCE_DIR}/serialization/IIR
#   DEPENDS ${iir_proto_python_files}
# )

