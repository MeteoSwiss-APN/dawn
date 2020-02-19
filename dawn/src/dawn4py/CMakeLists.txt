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

# Copy over to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY} PATTERN "*.py")

# Generate python proto files
include(DawnProtobufGenerate)

# Defines dawn_sir_proto_files and dawn_iir_proto_files
include(${PROJECT_SOURCE_DIR}/src/dawn/SIR/proto/DawnSIR.cmake)
include(${PROJECT_SOURCE_DIR}/src/dawn/IIR/proto/DawnIIR.cmake)

dawn_protobuf_generate(
  OUT_FILES sir_proto_python_files
  PROTOS ${dawn_sir_proto_files}
  WDIR ${PROJECT_SOURCE_DIR}/src/dawn/SIR/proto
  OUTDIR ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dawn4py/serialization
  PACKG SIR
  LANGUAGE python
)

dawn_protobuf_generate(
  OUT_FILES iir_proto_python_files
  PROTOS ${dawn_iir_proto_files}
  INC_DIR ${PROJECT_SOURCE_DIR}/src/dawn/SIR/proto
  WDIR ${PROJECT_SOURCE_DIR}/src/dawn/IIR/proto
  OUTDIR ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dawn4py/serialization
  PACKG IIR
  LANGUAGE python
)

add_custom_target(generate_python_proto DEPENDS ${sir_proto_python_files} ${iir_proto_python_files})

# Build python module
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(pybind11_fetch
  URL https://github.com/pybind/pybind11/archive/v2.4.0.tar.gz
)
FetchContent_GetProperties(pybind11_fetch)
if(NOT pybind11_fetch_POPULATED)
  if(NOT DAWN_INCLUDE_PYBIND11)
    message(STATUS "Fetching pybind11 for python bindings...")
  endif()
  FetchContent_Populate(pybind11_fetch)
  add_subdirectory(${pybind11_fetch_SOURCE_DIR} ${pybind11_fetch_BINARY_DIR})
  set(DAWN_INCLUDE_PYBIND11 ON CACHE BOOL "Include pybind11 when building Dawn python bindings.")
endif()

# Remove when updated past v2.4.3. See https://github.com/pybind/pybind11/issues/1604
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(pybind11 INTERFACE -fsized-deallocation)
endif()

pybind11_add_module(_dawn4py MODULE _dawn4py.cpp)
target_compile_features(_dawn4py PUBLIC cxx_std_17)
target_link_libraries(_dawn4py PUBLIC pybind11::pybind11 Dawn)
target_compile_options(_dawn4py PUBLIC -Wno-shadow)
#set_target_properties(_dawn4py PROPERTIES
#  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dawn4py)
add_dependencies(_dawn4py generate_python_proto)
