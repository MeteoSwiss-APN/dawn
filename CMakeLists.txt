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

if(NOT(DEFINED MSBUILD_ROOT))
    message(FATAL_ERROR "Specifying a MSBUILD_ROOT is required")
endif()
list(APPEND CMAKE_MODULE_PATH "${MSBUILD_ROOT}/cmake")

include(DawnCMakeInit)
set(DAWN_ROOT ${CMAKE_CURRENT_LIST_DIR})
dawn_cmake_init()

include(msbuildAddLibrary)
include(msbuildAddTargetClangFormat)
include(msbuildAddTargetCleanAll)
include(msbuildCheckInSourceBuild)
include(msbuildCombineLibraries)
include(msbuildConfigureFile)
include(msbuildCreatePackageString)
include(DawnExportPackage)
include(msbuildGetArchitectureInfo)
include(msbuildGetCompilerInfo)
include(msbuildGetGitHeadRevision)
include(msbuildGetPlatformInfo)
include(msbuildMakeStringPair)
include(msbuildReportResult)
include(msbuildSetCXXStandard)
include(DawnEnableFullRPATH)

msbuild_get_compiler_info()
msbuild_get_platform_info()
msbuild_get_architecture_info()

# Include the Dawn specific options, definitions and macros
include(DawnOptions)
include(DawnDefinitions)
include(DawnMacros)

# Set C++ standard
msbuild_set_cxx_standard(c++11)

# Set C++ flags
dawn_set_cxx_flags()

# Prevent in source builds
msbuild_check_in_source_build()

# Add custom targets
msbuild_add_target_clean_all(
  ${CMAKE_BINARY_DIR}/docs 
  ${CMAKE_BINARY_DIR}/sphinx
  ${CMAKE_BINARY_DIR}/src
  ${CMAKE_BINARY_DIR}/test
  ${CMAKE_BINARY_DIR}/bin
  ${CMAKE_BINARY_DIR}/python
  ${CMAKE_BINARY_DIR}/dawn-cmake
)

# Enable RPath support
dawn_enable_full_rpath(${DAWN_INSTALL_LIB_DIR})

# Output summary of the configuration
macro(make_config_string FIRST SECOND)
  msbuild_make_string_pair(${FIRST} ": ${SECOND}" 20 out)
  list(APPEND config_info ${out})
endmacro()
make_config_string("Dawn version" ${DAWN_FULL_VERSION_STR})
make_config_string("Platform" ${MSBUILD_PLATFORM_STRING})
make_config_string("Architecture" ${MSBUILD_ARCHITECTURE_STRING})
make_config_string("Compiler" ${MSBUILD_COMPILER_STRING})
make_config_string("Build type" ${CMAKE_BUILD_TYPE})
make_config_string("Asserts" ${DAWN_ASSERTS})
make_config_string("Build shared " ${BUILD_SHARED_LIBS})
make_config_string("Install prefix" ${CMAKE_INSTALL_PREFIX})
msbuild_report_result("Configuration summary" ${config_info})

# Include the packages 
if(DAWN_DOCUMENTATION)
  set(doc_packages Python3 Sphinx LaTeX Doxygen)
endif()

foreach(package bash Protobuf ccache clang-format ${doc_packages})
  include("Add${package}")
  msbuild_create_package_string(${package} info)
  list(APPEND package_info ${info})

    string(TOUPPER ${package} PACKAGE)
  list(APPEND DAW_EXTERNAL_LIBRARIES ${DAWN_${PACKAGE}_LIBRARIES})
  list(APPEND DAWN_EXTERNAL_INCLUDE_DIRS ${DAWN_${PACKAGE}_INCLUDE_DIRS})
  list(APPEND DAWN_EXTERNAL_DEFINITIONS ${DAWN_${PACKAGE}_DEFINITIONS})
endforeach()

include_directories(SYSTEM ${DAWN_EXTERNAL_INCLUDE_DIRS})
add_definitions(${DAWN_EXTERNAL_DEFINITIONS})

msbuild_report_result("Package summary" ${package_info})

# Add clang-format target
msbuild_add_target_clang_format(DIRECTORIES ${CMAKE_SOURCE_DIR}/src EXTENSION ".h;.cpp")

# Build dawn
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/src)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/test/utils/googletest/include)

add_subdirectory(src)
add_subdirectory(cmake)

if(DAWN_PYTHON)
  add_subdirectory(python)
endif()

if(DAWN_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(DAWN_DOCUMENTATION)
  add_subdirectory(docs)
endif()

# Install headers
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/src/ 
  DESTINATION ${DAWN_INSTALL_INCLUDE_DIR} 
  FILES_MATCHING PATTERN "*.h" PATTERN "*.inc" PATTERN "*.hpp"
)

# Install CMake package configuration
dawn_gen_install_config()
