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

project(dawn C CXX)
cmake_minimum_required(VERSION 3.3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(DawnCMakeInit)
set(DAWN_ROOT ${CMAKE_CURRENT_LIST_DIR})
dawn_cmake_init()

include(DawnAddLibrary)
include(DawnAddTargetClangFormat)
include(DawnAddTargetCleanAll)
include(DawnCheckInSourceBuild)
include(DawnCombineLibraries)
include(DawnConfigureFile)
include(DawnCreatePackageString)
include(DawnExportPackage)
include(DawnGetArchitectureInfo)
include(DawnGetCompilerInfo)
include(DawnGetGitHeadRevision)
include(DawnGetPlatformInfo)
include(DawnMakeStringPair)
include(DawnReportResult)
include(DawnSetCXXStandard)

dawn_get_compiler_info()
dawn_get_platform_info()
dawn_get_architecture_info()

# Include the Dawn specific options, definitions and macros
include(DawnOptions)
include(DawnDefinitions)
include(DawnMacros)

# Set C++ standard
dawn_set_cxx_standard(c++11)

# Set C++ flags
dawn_set_cxx_flags()

# Prevent in source builds
dawn_check_in_source_build()

# Add custom targets
dawn_add_target_clean_all(
  ${CMAKE_BINARY_DIR}/docs 
  ${CMAKE_BINARY_DIR}/sphinx
  ${CMAKE_BINARY_DIR}/src
  ${CMAKE_BINARY_DIR}/test
  ${CMAKE_BINARY_DIR}/bin
  ${CMAKE_BINARY_DIR}/dawn-cmake
)

# Output summary of the configuration
macro(make_config_string FIRST SECOND)
  dawn_make_string_pair(${FIRST} ": ${SECOND}" 20 out)
  list(APPEND config_info ${out})
endmacro()
make_config_string("Dawn version" ${DAWN_FULL_VERSION_STR})
make_config_string("Platform" ${DAWN_PLATFORM_STRING})
make_config_string("Architecture" ${DAWN_ARCHITECTURE_STRING})
make_config_string("Compiler" ${DAWN_COMPILER_STRING})
make_config_string("Build type" ${CMAKE_BUILD_TYPE})
make_config_string("Asserts" ${DAWN_ASSERTS})
make_config_string("Build shared " ${BUILD_SHARED_LIBS})
make_config_string("Install prefix" ${CMAKE_INSTALL_PREFIX})
dawn_report_result("Configuration summary" ${config_info})

# Include the packages 
if(DAWN_DOCUMENTATION)
  set(doc_packages bash Python3 Sphinx LaTeX)
endif()

foreach(package ccache clang-format ${doc_packages})
  include("Add${package}")
  dawn_create_package_string(${package} info)
  list(APPEND package_info ${info})
endforeach()
dawn_report_result("Package summary" ${package_info})

# Add clang-format target
dawn_add_target_clang_format(DIRECTORIES ${CMAKE_SOURCE_DIR}/src EXTENSION ".h;.cpp")

# Build dawn
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/src)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/test/utils/googletest/include)

add_subdirectory(src)

if(DAWN_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(DAWN_DOCUMENTATION)
  add_subdirectory(docs)
endif()

dawn_gen_install_config()
