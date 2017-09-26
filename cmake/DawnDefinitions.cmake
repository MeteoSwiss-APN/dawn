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

if(NOT DEFINED DAWN_VERSION_STRING)
  set(DAWN_VERSION_STRING 
      "${DAWN_VERSION_MAJOR}.${DAWN_VERSION_MINOR}.${DAWN_VERSION_PATCH}-${DAWN_VERSION_SUFFIX}"
      CACHE INTERNAL "Version string of Dawn" FORCE)
endif()

# Git version string
dawn_get_git_head_revision(git_refspec git_hash)
if(NOT DEFINED SDAWN_GIT_HASH OR NOT "${SDAWN_GIT_HASH}" STREQUAL "${git_hash}")
  string(SUBSTRING ${git_hash} 0 7 git_hash_short)
  set(SDAWN_GIT_HASH "${git_hash_short}" CACHE INTERNAL "git hash (short) of current head" FORCE)
endif()

# Assemble full version string
string(TOLOWER ${DAWN_ARCHITECTURE_STRING} architecture)
string(TOLOWER ${DAWN_PLATFORM_STRING} platform)
string(TOLOWER ${CMAKE_CXX_COMPILER_ID} compiler)
set(compiler "${compiler}-${CMAKE_CXX_COMPILER_VERSION}")
set(DAWN_FULL_VERSION_STRING 
    "${DAWN_VERSION_STRING}-${SDAWN_GIT_HASH}-${architecture}-${platform}-${compiler}"
    CACHE STRING "Full version string of Dawn" FORCE)
