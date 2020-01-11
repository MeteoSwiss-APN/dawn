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
target_link_libraries(_dawn4py PUBLIC pybind11::pybind11 Dawn)
target_compile_options(_dawn4py PUBLIC -Wno-shadow)

# Depending on DawnSIRProto and DawnIIRProto ensures the python files are
# generated and moved into the correct location
# TODO Do we need to depend on Dawn or DawnC here?
add_custom_target(python DEPENDS DawnSIRProto DawnIIRProto _dawn4py)
