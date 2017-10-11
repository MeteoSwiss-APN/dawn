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

# Version
if(NOT DEFINED GTCLANG_VERSION_MAJOR)
  set(GTCLANG_VERSION_MAJOR 0 CACHE INTERNAL "Major version of gtclang" FORCE)
endif()

if(NOT DEFINED GTCLANG_VERSION_MINOR)
  set(GTCLANG_VERSION_MINOR 0 CACHE INTERNAL "Minor version of gtclang" FORCE)
endif()

if(NOT DEFINED GTCLANG_VERSION_PATCH)
  set(GTCLANG_VERSION_PATCH 1 CACHE INTERNAL "Patch version of gtcang" FORCE)
endif()

if(NOT DEFINED GTCLANG_VERSION)
  set(GTCLANG_VERSION_STRING 
      ${GTCLANG_VERSION_MAJOR}.${GTCLANG_VERSION_MINOR}.${GTCLANG_VERSION_PATCH}
      CACHE INTERNAL "Version of gtclang" FORCE)
endif()

# Git version string
dawn_get_git_head_revision(git_refspec git_hash)
if(NOT DEFINED SGTCLANG_GIT_HASH OR NOT "${SGTCLANG_GIT_HASH}" STREQUAL "${git_hash}")
  string(SUBSTRING ${git_hash} 0 7 git_hash_short)
  set(SGTCLANG_GIT_HASH "${git_hash_short}" CACHE INTERNAL "git hash (short) of current head" FORCE)
endif()

# Assemble full version string
string(TOLOWER ${DAWN_ARCHITECTURE_STRING} architecture)
string(TOLOWER ${DAWN_PLATFORM_STRING} platform)
string(TOLOWER ${CMAKE_CXX_COMPILER_ID} compiler)
set(compiler "${compiler}-${CMAKE_CXX_COMPILER_VERSION}")
set(GTCLANG_FULL_VERSION 
    "${GTCLANG_VERSION_STRING}-${SGTCLANG_GIT_HASH}-${architecture}-${platform}-${compiler}"
    CACHE STRING "Full version string of gtclang" FORCE)

# Building configs
set(GTCLANG_DSL_INCLUDES "${CMAKE_SOURCE_DIR}/src" "${CMAKE_INSTALL_PREFIX}/include")

# Testing configs
set(GTCLANG_EXECUTABLE "${CMAKE_BINARY_DIR}/bin/gtclang")
set(GTCLANG_UNITTEST_DATAPATH ${CMAKE_SOURCE_DIR}/test/integration)
set(GTCLANG_UNITTEST_INCLUDES ${CMAKE_SOURCE_DIR}/src)

# Installation definitions
set(GTCLANG_INSTALL_BIN_DIR bin 
    CACHE INTERNAL "Relative path of the binary install location" FORCE)
set(GTCLANG_INSTALL_INCLUDE_DIR include 
    CACHE INTERNAL "Relative path of the include install location" FORCE)
set(GTCLANG_INSTALL_LIB_DIR lib 
    CACHE INTERNAL "Relative path of the library install location " FORCE)
set(GTCLANG_INSTALL_CMAKE_DIR cmake 
    CACHE INTERNAL "Relative path of the cmake install location" FORCE)
