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

set(DAWN_YODA_GIT_URL "https:///github.com/MeteoSwiss-APN/yoda.git"
    CACHE PATH "URL of the dawn git repository to clone")
set(DAWN_YODA_GIT_BRANCH "1.0.10" CACHE STRING "Branch of the dawn git repository to clone")
dawn_clone_repository(NAME yoda URL ${DAWN_YODA_GIT_URL} BRANCH ${DAWN_YODA_GIT_BRANCH} SOURCE_DIR DAWN_YODA_SOURCE_DIR )
#
list(APPEND CMAKE_MODULE_PATH "${DAWN_YODA_SOURCE_DIR}/cmake")
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
  "${CMAKE_BINARY_DIR}/Makefile"
  "${CMAKE_BINARY_DIR}/yoda-cmake"
  "${CMAKE_BINARY_DIR}/protobuf"
  "${CMAKE_BINARY_DIR}/protobuf-prefix"
)

option(DAWN_BUNDLE_PYTHON "Build and install the Python module interface to HIR" OFF)
option(DAWN_BUNDLE_JAVA "Build and install the java interface to HIR" OFF)
option(DAWN_PYTHON_EXAMPLES "Build the python examples of generating HIR" OFF)

if( DAWN_BUNDLE_PYTHON )
  if(NOT DEFINED PYTHON_EXECUTABLE)
    find_package(PythonInterp 3 REQUIRED)
  endif()
endif()

include("thirdparty/DawnAddProtobuf")

set(dawn_cmake_args -DProtobuf_DIR=${Protobuf_DIR} -DDAWN_PYTHON=${DAWN_BUNDLE_PYTHON} -DDAWN_EXAMPLES=${DAWN_PYTHON_EXAMPLES} -DDAWN_JAVA=${DAWN_BUNDLE_JAVA})

yoda_find_package(
  PACKAGE dawn
  FORWARD_VARS
    BINARY_DIR dawn_binary_dir
  DEPENDS "protobuf"
  ADDITIONAL
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/../"
    YODA_ROOT "${DAWN_YODA_SOURCE_DIR}"
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

if(DAWN_BUNDLE_PYTHON)
  if(NOT DEFINED PROTOBUF_PYTHON_INSTALL)
    message(FATAL_ERROR "protobuf python not set")
  endif()
  install(DIRECTORY ${PROTOBUF_PYTHON_INSTALL}/google DESTINATION ${CMAKE_INSTALL_PREFIX}/python)
  install (SCRIPT "${CMAKE_SOURCE_DIR}/PostInstall.cmake")
endif(DAWN_BUNDLE_PYTHON)
