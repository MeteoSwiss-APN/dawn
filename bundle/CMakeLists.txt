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
enable_testing()
cmake_minimum_required(VERSION 3.8.1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")

include(DawnCloneRepository)

set(DAWN_MCHBUILD_GIT_URL "https:///github.com/cosunae/msbuild.git"
    CACHE PATH "URL of the dawn git repository to clone")
set(DAWN_MCHBUILD_GIT_BRANCH "master" CACHE STRING "Branch of the dawn git repository to clone")
message("HOHOHOHOHO")
dawn_clone_repository(NAME mchbuild URL ${DAWN_MCHBUILD_GIT_URL} BRANCH ${DAWN_MCHBUILD_GIT_BRANCH} SOURCE_DIR DAWN_MCHBUILD_SOURCE_DIR )
message("KKKKK ${DAWN_MCHBUILD_SOURCE_DIR}")
#
list(APPEND CMAKE_MODULE_PATH "${DAWN_MCHBUILD_SOURCE_DIR}/cmake")
include(mchbuildCMakeInit)

include(mchbuildAddTargetCleanAll)
include(mchbuildCheckInSourceBuild)
include(mchbuildGetArchitectureInfo)
include(mchbuildGetCompilerInfo)
include(mchbuildGetPlatformInfo)
include(mchbuildMakeStringPair)
include(mchbuildReportResult)

mchbuild_get_compiler_info()
mchbuild_get_platform_info()
mchbuild_get_architecture_info()

# Prevent in source builds
mchbuild_check_in_source_build()

# Add custom targets
mchbuild_add_target_clean_all(
  "${CMAKE_BINARY_DIR}/prefix"
  "${CMAKE_BINARY_DIR}/thirdparty"  
  "${CMAKE_BINARY_DIR}/dawn-cmake"  
  "${CMAKE_BINARY_DIR}/dawn"
  "${CMAKE_BINARY_DIR}/dawn-prefix"
  "${CMAKE_BINARY_DIR}/gtclang"
  "${CMAKE_BINARY_DIR}/gtclang-prefix"
  "${CMAKE_BINARY_DIR}/Makefile"
  "${CMAKE_BINARY_DIR}/mchbuild-cmake"
  "${CMAKE_BINARY_DIR}/protobuf"
  "${CMAKE_BINARY_DIR}/protobuf-prefix"
)

# Output summary of the configuration
macro(make_config_string FIRST SECOND)
  mchbuild_make_string_pair(${FIRST} ": ${SECOND}" 20 out)
  list(APPEND config_info ${out})
endmacro()
make_config_string("Platform" ${MCHBUILD_PLATFORM_STRING})
make_config_string("Architecture" ${MCHBUILD_ARCHITECTURE_STRING})
make_config_string("Compiler" ${MCHBUILD_COMPILER_STRING})
make_config_string("Build type" ${CMAKE_BUILD_TYPE})
mchbuild_report_result("Configuration summary" ${config_info})

include("thirdparty/DawnAddProtobuf")

set(dawn_cmake_args -DProtobuf_DIR=${Protobuf_DIR})

mchbuild_find_package(
  PACKAGE dawn
  FORWARD_VARS 
    BINARY_DIR dawn_binary_dir
  DEPENDS "protobuf"
  ADDITIONAL
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/../"
    MCHBUILD_ROOT "${DAWN_MCHBUILD_SOURCE_DIR}"
    CMAKE_ARGS 
        ${dawn_cmake_args}
)

if(NOT(dawn_binary_dir))
  message(FATAL_ERROR "dawn binary dir was not defined by External_dawn")
endif()

add_test(NAME dawn-tests 
  COMMAND  ${CMAKE_COMMAND} --build ${dawn_binary_dir} --target test
)

set(conf_bundle_filename ${CMAKE_BINARY_DIR}/dawn-conf.bundle)
file(WRITE ${conf_bundle_filename} "conf")

install(
  FILES ${conf_bundle_filename}
  DESTINATION bundle_conf
)

