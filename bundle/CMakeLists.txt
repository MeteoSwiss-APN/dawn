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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")

include(GTClangCloneRepository)

set(GTCLANG_MCHBUILD_GIT_URL "https:///github.com/Meteoswiss-APN/mchbuild.git" 
    CACHE PATH "URL of the mchbuild git repository to clone")
set(GTCLANG_MCHBUILD_GIT_BRANCH "master" CACHE STRING "Branch of the mchbuild git repository to clone")

gtclang_clone_repository(NAME mchbuild URL ${GTCLANG_MCHBUILD_GIT_URL} BRANCH ${GTCLANG_MCHBUILD_GIT_BRANCH} SOURCE_DIR GTCLANG_MCHBUILD_SOURCE_DIR )

list(APPEND CMAKE_MODULE_PATH "${GTCLANG_MCHBUILD_SOURCE_DIR}/cmake")
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
  "${CMAKE_BINARY_DIR}/mchbuild-cmake"
  "${CMAKE_BINARY_DIR}/boost"
  "${CMAKE_BINARY_DIR}/boost-prefix"
)

include(GTClangAddDependency)
include(mchbuildFindPackage)
include(mchbuildSetDownloadDir)

# Set the default download directory (define MCHBUILD_DOWNLOAD_DIR)
mchbuild_set_download_dir()

include("thirdparty/gtclangAddBoost")
set(dawn_cmake_args -DBOOST_ROOT=${BOOST_ROOT} -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>)

mchbuild_find_package(
  PACKAGE dawn
  NO_DEFAULT_PATH
  REQUIRED_VARS dawn_DIR
  ADDITIONAL
    DOWNLOAD_DIR ${MCHBUILD_DOWNLOAD_DIR}
    GIT_REPOSITORY "git@github.com:Meteoswiss-APN/dawn.git"
    GIT_TAG "master" 
    MCHBUILD_ROOT "${GTCLANG_MCHBUILD_SOURCE_DIR}"
    CMAKE_ARGS ${dawn_cmake_args}
)


#TODO Cleanup the mess with the naming conventions
set(gtclang_cmake_args -Ddawn_DIR=${dawn_DIR} -DBOOST_ROOT=${BOOST_ROOT})
option(GTCLANG_ENABLE_GRIDTOOLS "Enable GridTools compilation" OFF)

set(gtclang_dependencies "boost" "dawn")

if(GTCLANG_ENABLE_GRIDTOOLS) 
  mchbuild_find_package(
    PACKAGE GridTools
    REQUIRED_VARS GridTools_DIR
    DEPENDS "boost"
    ADDITIONAL
      DOWNLOAD_DIR ${MCHBUILD_DOWNLOAD_DIR}
      GIT_REPOSITORY "git@github.com:twicki/gridtools.git"
      GIT_TAG "gtclang_fixed_version" 
      MCHBUILD_ROOT "${GTCLANG_MCHBUILD_SOURCE_DIR}"
      CMAKE_ARGS -DSTRUCTURED_GRIDS=ON -DBOOST_ROOT=${BOOST_ROOT} -DDISABLE_TESTING=ON -DENABLE_PERFORMANCE_METERS=ON
  )
  list(APPEND gtclang_dependencies "gridtools")

  list(APPEND gtclang_cmake_args -DGridTools_DIR=${GridTools_DIR})
else()
  list(APPEND gtclang_cmake_args -DGTCLANG_BUILD_EXAMPLES=OFF)
endif()

mchbuild_find_package(
  PACKAGE gtclang
  NO_DEFAULT_PATH
  FORWARD_VARS 
    BINARY_DIR gtclang_binary_dir
  DEPENDS ${gtclang_dependencies}
  ADDITIONAL
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/../"
    MCHBUILD_ROOT "${GTCLANG_MCHBUILD_SOURCE_DIR}"
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

