##===------------------------------------------------------------------------------*- CMake -*-===##
##                         _       _
##                        | |     | |
##                    __ _| |_ ___| | __ _ _ __   __ _ 
##                   / _` | __/ __| |/ _` | '_ \ / _` |
##                  | (_| | || (__| | (_| | | | | (_| |
##                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
##                    __/ |                       __/ |
##                   |___/                       |___/
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
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")

include(DawnCloneRepository)

set(MB_GIT_URL "git@github.com:cosunae/msbuild.git" 
    CACHE PATH "URL of the dawn git repository to clone")
set(MB_GIT_BRANCH "master" CACHE STRING "Branch of the dawn git repository to clone")

dawn_clone_repository(NAME msbuild URL ${MB_GIT_URL} BRANCH ${MB_GIT_BRANCH})

#
list(APPEND CMAKE_MODULE_PATH "${DAWN_MSBUILD_SOURCE_DIR}/cmake")
include(msbuildCMakeInit)

include(msbuildAddTargetCleanAll)
include(msbuildCheckInSourceBuild)
include(msbuildGetArchitectureInfo)
include(msbuildGetCompilerInfo)
include(msbuildGetPlatformInfo)
include(msbuildMakeStringPair)
include(msbuildReportResult)

# Include all options (including all Dawn and GTClang options)
#include(GTClangAllOptions)

msbuild_get_compiler_info()
msbuild_get_platform_info()
msbuild_get_architecture_info()

# Prevent in source builds
msbuild_check_in_source_build()

# Add custom targets
msbuild_add_target_clean_all(
  "${CMAKE_BINARY_DIR}/prefix"
  "${CMAKE_BINARY_DIR}/thirdparty"  
  "${CMAKE_BINARY_DIR}/dawn-cmake"  
  "${CMAKE_BINARY_DIR}/dawn"
  "${CMAKE_BINARY_DIR}/dawn-prefix"
  "${CMAKE_BINARY_DIR}/gtclang"
  "${CMAKE_BINARY_DIR}/gtclang-prefix"
  "${CMAKE_BINARY_DIR}/Makefile"
  "${CMAKE_BINARY_DIR}/msbuild-cmake"
  "${CMAKE_BINARY_DIR}/protobuf"
  "${CMAKE_BINARY_DIR}/protobuf-prefix"
)

# Output summary of the configuration
macro(make_config_string FIRST SECOND)
  msbuild_make_string_pair(${FIRST} ": ${SECOND}" 20 out)
  list(APPEND config_info ${out})
endmacro()
make_config_string("Platform" ${MSBUILD_PLATFORM_STRING})
make_config_string("Architecture" ${MSBUILD_ARCHITECTURE_STRING})
make_config_string("Compiler" ${MSBUILD_COMPILER_STRING})
make_config_string("Build type" ${CMAKE_BUILD_TYPE})
msbuild_report_result("Configuration summary" ${config_info})

include(DawnAddDependency)

dawn_add_dependency(NAME "protobuf")


set(dawn_cmake_args -DProtobuf_DIR=${Protobuf_DIR})

msbuild_find_package(
  PACKAGE dawn
  DEPENDS "protobuf"
  ADDITIONAL
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/../"
    MSBUILD_ROOT "${DAWN_MSBUILD_SOURCE_DIR}"
    CMAKE_ARGS 
        ${dawn_cmake_args}
)

file(WRITE ${CMAKE_BINARY_DIR}/Conf.bundle "conf")

install(
  FILES ${CMAKE_BINARY_DIR}/Conf.bundle
  DESTINATION bundle_conf
)

