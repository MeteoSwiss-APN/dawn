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

# Version
if(NOT DEFINED DAWN_VERSION_MAJOR)
  set(DAWN_VERSION_MAJOR 0 CACHE INTERNAL "Major version of Dawn" FORCE)
endif()

if(NOT DEFINED DAWN_VERSION_MINOR)
  set(DAWN_VERSION_MINOR 0 CACHE INTERNAL "Minor version of Dawn" FORCE)
endif()

if(NOT DEFINED DAWN_VERSION_PATCH)
  set(DAWN_VERSION_PATCH 1 CACHE INTERNAL "Patch version of gtcang" FORCE)
endif()

if(NOT DEFINED DAWN_VERSION_SUFFIX)
  set(DAWN_VERSION_SUFFIX "dev" CACHE INTERNAL "Suffix of the Dawn version" FORCE)
endif()

set(DAWN_VERSION ${DAWN_VERSION_MAJOR}.${DAWN_VERSION_MINOR}.${DAWN_VERSION_PATCH}
    CACHE STRING "Version of Dawn" FORCE)

mark_as_advanced(DAWN_VERSION)

# Git version string
dawn_get_git_head_revision(git_refspec git_hash)
string(SUBSTRING "${git_hash}" 0 7 git_hash_short)
if(NOT DEFINED DAWN_GIT_HASH OR NOT "${DAWN_GIT_HASH}" STREQUAL "${git_hash_short}")
  set(DAWN_GIT_HASH "${git_hash_short}" CACHE INTERNAL "git hash (short) of current head" FORCE)
endif()

# Assemble full version string
string(TOLOWER ${MSBUILD_ARCHITECTURE_STRING} architecture)
string(TOLOWER ${MSBUILD_PLATFORM_STRING} platform)
string(TOLOWER ${CMAKE_CXX_COMPILER_ID} compiler)
set(compiler "${compiler}-${CMAKE_CXX_COMPILER_VERSION}")
set(DAWN_FULL_VERSION_STR 
    "${DAWN_VERSION}-${SDAWN_GIT_HASH}-${architecture}-${platform}-${compiler}"
    CACHE STRING "Full version string of Dawn" FORCE)

mark_as_advanced(DAWN_FULL_VERSION_STR)

# Installation definitions
set(DAWN_INSTALL_INCLUDE_DIR include 
    CACHE INTERNAL "Relative path of the include install location" FORCE)
set(DAWN_INSTALL_LIB_DIR lib 
    CACHE INTERNAL "Relative path of the library install location " FORCE)
set(DAWN_INSTALL_CMAKE_DIR cmake 
    CACHE INTERNAL "Relative path of the cmake install location" FORCE)
set(DAWN_INSTALL_PYTHON_DIR python 
    CACHE INTERNAL "Relative path of the cmake install location" FORCE)

