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

set(DAWN_MCHBUILD_GIT_URL "https:///github.com/Meteoswiss-APN/mchbuild.git"
    CACHE PATH "URL of the dawn git repository to clone")
set(DAWN_MCHBUILD_GIT_BRANCH "master" CACHE STRING "Branch of the dawn git repository to clone")
dawn_clone_repository(NAME mchbuild URL ${DAWN_MCHBUILD_GIT_URL} BRANCH ${DAWN_MCHBUILD_GIT_BRANCH} SOURCE_DIR DAWN_MCHBUILD_SOURCE_DIR )
#
list(APPEND CMAKE_MODULE_PATH "${DAWN_MCHBUILD_SOURCE_DIR}/cmake")
include(mchbuildInit)

mchbuild_init()

include(mchbuildAddTargetCleanAll)

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

include("thirdparty/DawnAddProtobuf")
include("thirdparty/DawnAddBoost")

set(dawn_cmake_args -DProtobuf_DIR=${Protobuf_DIR} -DBOOST_ROOT=${BOOST_ROOT})

mchbuild_find_package(
  PACKAGE dawn
  FORWARD_VARS 
    BINARY_DIR dawn_binary_dir
  DEPENDS "protobuf" "boost"
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

