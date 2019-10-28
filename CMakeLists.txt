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

if(NOT BUILD_SHARED_LIBS)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries." FORCE)
endif()

cmake_minimum_required(VERSION 3.3)
project(dawn
    VERSION 0.0.1
    DESCRIPTION "Dawn is a compiler toolchain for developing high-level DSLs for geophysical fluid dynamics models"
    HOMEPAGE_URL https://meteoswiss-apn.github.io/dawn/
    LANGUAGES C CXX)

option(GTCLANG_BUILD_GT_CPU_EXAMPLES "Build GridTools CPU examples. Requires GTCLANG_ENABLE_GRIDTOOLS=ON." OFF)
option(GTCLANG_BUILD_GT_CPU_EXAMPLES "Build GridTools GPU examples. Requires GTCLANG_ENABLE_GRIDTOOLS=ON." OFF)
option(GTCLANG_BUILD_CUDA_EXAMPLES "Build CUDA GPU examples." OFF)
option(DAWN_PYTHON "Enable python bindings for dawn. Requires the google.protobuf python package." ON)
set(PYTHON_PROTOBUF_DIR "/usr/lib/python3/dist-packages" CACHE PATH "Path to the google protobuf python package.")

set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH
  "Path prefix prepended onto install directories." FORCE)
endif()

set(YODA_ROOT CACHE PATH "path to yoda package")

if("${YODA_ROOT}" STREQUAL "")
    message(FATAL_ERROR "YODA_ROOT not found! Try specifying it in the environment via -DYODA_ROOT=<>")
endif()

set(DAWN_SOURCE_DIR "${CMAKE_SOURCE_DIR}/dawn")
set(GTCLANG_SOURCE_DIR "${CMAKE_SOURCE_DIR}/gtclang")

list(APPEND CMAKE_MODULE_PATH "${YODA_ROOT}/cmake")
list(APPEND CMAKE_MODULE_PATH "${DAWN_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${GTCLANG_SOURCE_DIR}/cmake")

include(yodaMakeStringPair)
include(yodaReportResult)
include(yodaAddExecutable)
include(yodaAddLibrary)
include(yodaAddTargetCleanAll)
include(yodaAddTargetClangFormat)
include(yodaCombineLibraries)
include(yodaConfigureFile)
include(yodaCreatePackageString)
include(yodaEnableFullRPATH)
include(yodaExportPackage)
include(yodaGetGitHeadRevision)
include(yodaSetCXXStandard)
include(yodaAddCustomDummyTarget)
include(yodaInit)

yoda_init()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add custom targets
yoda_add_target_clean_all(
  ${CMAKE_BINARY_DIR}/docs
  ${CMAKE_BINARY_DIR}/sphinx
  ${CMAKE_BINARY_DIR}/src
  ${CMAKE_BINARY_DIR}/test
  ${CMAKE_BINARY_DIR}/bin
  ${CMAKE_BINARY_DIR}/python
  ${CMAKE_BINARY_DIR}/dawn-cmake
)

add_subdirectory(utils)
add_subdirectory(dawn)
add_subdirectory(gtclang)

# Add clang-format target
set(format_directories
  "${DAWN_SOURCE_DIR}/src"
  "${DAWN_SOURCE_DIR}/test"
  "${GTCLANG_SOURCE_DIR}/src"
  "${GTCLANG_SOURCE_DIR}/test/unit-test"
)

yoda_add_target_clang_format(DIRECTORIES ${format_directories} EXTENSION ".h;.cpp")
