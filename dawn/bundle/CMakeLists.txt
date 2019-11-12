##===--------------------------------------------------------------------------------*- C++ -*-===##
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

# This file builds dependencies for dawn. It is optionally used in place of
# manually installing these packages.

# NOTE: These options affect the _dependencies_, not dawn itself.

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo." FORCE)
endif()

cmake_minimum_required(VERSION 3.8.1)
project(dawn-bundle
  LANGUAGES C CXX)

option(DAWN_PYTHON "Installs protobuf python bindings for use in dawn." ON)

# Require Python for protobuf bindings
if(DAWN_PYTHON)
  find_package(Python3 COMPONENTS Interpreter)
  set(protobuf_step_targets python-build python-install)
endif()

include(ExternalProject)

# Dependency: Protobuf
# C++ protobuf
set(protobuf_version "3.4.0")
set(protobuf_version_short "3.4")

ExternalProject_Add(protobuf
  URL https://github.com/google/protobuf/archive/v${protobuf_version}.tar.gz
  URL_HASH MD5=1d077a7d4db3d75681f5c333f2de9b1a
  DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/downloads/"
  PREFIX "${CMAKE_BINARY_DIR}/protobuf-prefix"
  SOURCE_SUBDIR cmake
  CMAKE_ARGS
    "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
    "-Dprotobuf_BUILD_EXAMPLES=OFF"
    "-Dprotobuf_BUILD_SHARED_LIBS=OFF"
    "-Dprotobuf_BUILD_TESTS=OFF"
    "-Dprotobuf_INSTALL_EXAMPLES=OFF"
  STEP_TARGETS "${protobuf_step_targets}"
)


ExternalProject_Add_Step(
  protobuf python-build
  COMMAND PROTOC=<INSTALL_DIR>/bin/protoc ${Python3_EXECUTABLE} setup.py build
  WORKING_DIRECTORY "<SOURCE_DIR>/python"
  DEPENDEES build install
)

ExternalProject_Add_Step(
  protobuf python-install
  COMMAND ${CMAKE_COMMAND} -E copy_directory "<SOURCE_DIR>/python" "<INSTALL_DIR>/python"
  DEPENDEES build install python-build
)
