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

#
# Version
#
if(NOT DEFINED GTCLANG_VERSION_MAJOR)
  set(GTCLANG_VERSION_MAJOR 0 CACHE INTERNAL "Major version of GTCLANG" FORCE)
endif()

if(NOT DEFINED GTCLANG_VERSION_MINOR)
  set(GTCLANG_VERSION_MINOR 0 CACHE INTERNAL "Minor version of GTCLANG" FORCE)
endif()

if(NOT DEFINED GTCLANG_VERSION_PATCH)
  set(GTCLANG_VERSION_PATCH 1 CACHE INTERNAL "Patch version of GTCLANG" FORCE)
endif()

if(NOT DEFINED GTCLANG_VERSION_SUFFIX)
  set(GTCLANG_VERSION_SUFFIX "dev" CACHE INTERNAL "Suffix of the version" FORCE)
endif()

if(NOT DEFINED GTCLANG_VERSION_STRING)
  set(GTCLANG_VERSION_STRING 
      "${GTCLANG_VERSION_MAJOR}.${GTCLANG_VERSION_MINOR}.${GTCLANG_VERSION_PATCH}-${GTCLANG_VERSION_SUFFIX}"
      CACHE INTERNAL "Version string of GTCLANG" FORCE)
endif()

#
# Platform
#
if(APPLE)
  set(GTCLANG_ON_UNIX 1 CACHE INTERNAL "Platform is Unix-ish" FORCE)
  set(GTCLANG_ON_APPLE 1 CACHE INTERNAL "Platform is Darwin" FORCE)
  set(GTCLANG_PLATFORM_STRING "Darwin" CACHE INTERNAL "Platform-id string" FORCE)
elseif(UNIX)
  set(GTCLANG_ON_UNIX 1 CACHE INTERNAL "Platform is Unix-ish" FORCE)
  set(GTCLANG_ON_LINUX 1 CACHE INTERNAL "Platform is Linux" FORCE)
  set(GTCLANG_PLATFORM_STRING "Linux" CACHE INTERNAL "Platform-id string" FORCE)
endif()

#
# Compiler
#
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(GTCLANG_COMPILER_GNU 1 CACHE INTERNAL "Compiler is GNU gcc" FORCE)
  set(GTCLANG_COMPILER_STRING "GNU gcc (${CMAKE_CXX_COMPILER_VERSION})" 
      CACHE INTERNAL "Compiler-id string" FORCE)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(GTCLANG_COMPILER_CLANG 1 CACHE INTERNAL "Compiler is LLVM Clang" FORCE)
  set(GTCLANG_COMPILER_STRING "LLVM Clang (${CMAKE_CXX_COMPILER_VERSION})" 
      CACHE INTERNAL "Compiler-id string" FORCE)
endif()

#
# gridtools clang DSL headers
#
set(GTCLANG_DSL_INCLUDES "${CMAKE_SOURCE_DIR}/src" 
                         "${CMAKE_INSTALL_PREFIX}/include")

#
# Unittest
#
set(GTCLANG_EXECUTABLE "${CMAKE_BINARY_DIR}/bin/gtclang")
set(GTCLANG_UNITTEST_DATAPATH ${CMAKE_SOURCE_DIR}/test/integration)
set(GTCLANG_UNITTEST_INCLUDES ${CMAKE_SOURCE_DIR}/src)

