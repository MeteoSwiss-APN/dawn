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

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo." FORCE)
endif()

if(NOT BUILD_SHARED_LIBS)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries." FORCE)
endif()

if(NOT CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE STRING
    "Install path prefix, prepended onto install directories." FORCE)
endif()

cmake_minimum_required(VERSION 3.3)
project(dawn
    VERSION 0.0.1
    DESCRIPTION "Dawn is a compiler toolchain for developing high-level DSLs for geophysical fluid dynamics models"
    HOMEPAGE_URL https://meteoswiss-apn.github.io/dawn/
    LANGUAGES C CXX)

set(YODA_ROOT CACHE PATH "path to yoda package")

if("${YODA_ROOT}" STREQUAL "")
    message(FATAL_ERROR "YODA_ROOT not found! Try specifying it in the environment via -DYODA_ROOT=<>")
endif()

list(APPEND CMAKE_MODULE_PATH "${YODA_ROOT}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/dawn/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/gtclang/cmake")

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

add_subdirectory(dawn)
add_subdirectory(gtclang)