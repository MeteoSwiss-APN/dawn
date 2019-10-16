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

project(gtclang-all C CXX)
enable_testing()
cmake_minimum_required(VERSION 3.8.1)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.3)
        message(FATAL_ERROR "GCC version must be at least 5.3!")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message(FATAL_ERROR "Clang version must be at least 6.0!")
    endif()
else()
    message(WARNING "You are using an unsupported compiler! Compilation has only been tested with Clang and GCC.")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")

include(GTClangCloneRepository)

set(GTCLANG_YODA_GIT_URL "https:///github.com/MeteoSwiss-APN/yoda.git"
    CACHE PATH "URL of the yoda git repository to clone")
set(GTCLANG_YODA_GIT_BRANCH "1.0.9" CACHE STRING "Branch of the yoda git repository to clone")

gtclang_clone_repository(NAME yoda URL ${GTCLANG_YODA_GIT_URL} BRANCH ${GTCLANG_YODA_GIT_BRANCH} SOURCE_DIR GTCLANG_YODA_SOURCE_DIR )

list(APPEND CMAKE_MODULE_PATH "${GTCLANG_YODA_SOURCE_DIR}/cmake")
include(yodaInit)

yoda_init()

include(yodaAddTargetCleanAll)

# Add custom targets
yoda_add_target_clean_all(
  "${CMAKE_BINARY_DIR}/prefix"
  "${CMAKE_BINARY_DIR}/thirdparty"
  "${CMAKE_BINARY_DIR}/dawn-cmake"
  "${CMAKE_BINARY_DIR}/dawn"
  "${CMAKE_BINARY_DIR}/dawn-prefix"
  "${CMAKE_BINARY_DIR}/gtclang"
  "${CMAKE_BINARY_DIR}/gtclang-prefix"
  "${CMAKE_BINARY_DIR}/yoda-cmake"
  "${CMAKE_BINARY_DIR}/boost"
  "${CMAKE_BINARY_DIR}/boost-prefix"
)

include(yodaFindPackage)
include(yodaSetDownloadDir)

# Set the default download directory (define YODA_DOWNLOAD_DIR)
yoda_set_download_dir()

include("thirdparty/gtclangAddBoost")
set(dawn_cmake_args -DBOOST_ROOT=${BOOST_ROOT} -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>)

yoda_find_package(
  PACKAGE dawn
  NO_DEFAULT_PATH
  REQUIRED_VARS dawn_DIR
  ADDITIONAL
    DOWNLOAD_DIR ${YODA_DOWNLOAD_DIR}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../dawn/bundle
    YODA_ROOT "${GTCLANG_YODA_SOURCE_DIR}"
    CMAKE_ARGS ${dawn_cmake_args}
)


#TODO Cleanup the mess with the naming conventions
set(gtclang_cmake_args -Ddawn_DIR=${dawn_DIR} -DBOOST_ROOT=${BOOST_ROOT})
option(GTCLANG_ENABLE_GRIDTOOLS "Enable GridTools compilation" OFF)

set(gtclang_dependencies "boost" "dawn")

if(GTCLANG_ENABLE_GRIDTOOLS)
  yoda_find_package(
    PACKAGE GridTools
    REQUIRED_VARS GridTools_DIR
    DEPENDS "boost"
    ADDITIONAL
      DOWNLOAD_DIR ${YODA_DOWNLOAD_DIR}
      GIT_REPOSITORY "git@github.com:GridTools/gridtools.git"
      GIT_TAG "v1.0.3"
      YODA_ROOT "${GTCLANG_YODA_SOURCE_DIR}"
      CMAKE_ARGS -DSTRUCTURED_GRIDS=ON -DBOOST_ROOT=${BOOST_ROOT} -DBUILD_TESTING=OFF -DGT_ENABLE_PERFORMANCE_METERS=ON -DINSTALL_GTEST=OFF -DGT_USE_MPI=OFF
  )
  list(APPEND gtclang_dependencies "gridtools")

  list(APPEND gtclang_cmake_args -DGridTools_DIR=${GridTools_DIR})
else()
  list(APPEND gtclang_cmake_args -DGTCLANG_BUILD_GT_CPU_EXAMPLES=OFF)
  list(APPEND gtclang_cmake_args -DGTCLANG_BUILD_GT_GPU_EXAMPLES=OFF)
  list(APPEND gtclang_cmake_args -DGTCLANG_BUILD_CUDA_EXAMPLES=OFF)
endif()

yoda_find_package(
  PACKAGE gtclang
  NO_DEFAULT_PATH
  FORWARD_VARS
    BINARY_DIR gtclang_binary_dir
  DEPENDS ${gtclang_dependencies}
  ADDITIONAL
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/../"
    YODA_ROOT "${GTCLANG_YODA_SOURCE_DIR}"
    CMAKE_ARGS
        ${gtclang_cmake_args}
)

if(NOT(gtclang_binary_dir))
  message(FATAL_ERROR "gtclang binary dir was not defined by External_gtclang")
endif()

add_test(NAME gtclang-tests
  COMMAND  ${CMAKE_COMMAND} --build ${gtclang_binary_dir} --target test
)

#We need to install something
set(conf_bundle_filename ${CMAKE_BINARY_DIR}/gtclang-conf.bundle)

file(WRITE ${conf_bundle_filename} "conf")

install(
  FILES ${conf_bundle_filename}
  DESTINATION bundle_conf
)
