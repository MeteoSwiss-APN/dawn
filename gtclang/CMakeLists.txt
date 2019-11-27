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

cmake_minimum_required(VERSION 3.13)

# project version
set(VERSION_MAJOR 0)
set(VERSION_MINOR 0)
set(VERSION_PATCH 1)

set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

project(GTClang
  VERSION ${VERSION}
  HOMEPAGE_URL https://meteoswiss-apn.github.io/dawn/
  LANGUAGES CXX C)

# Useful standard options:
# - BUILD_TESTING<Bool>: Build test target (if only project, otherwise use GTCLANG_BUILD_TESTING to override).
# - BUILD_SHARED_LIBS<Bool>: Build gtclang shared libraries instead of static. If true, requires dependencies to be built as position independent code
# - CMAKE_PREFIX_PATH<String>: Semi-colon separated path to dependencies.

# Other options
include(CMakeDependentOption)
option(BUILD_EXAMPLES "Build examples" ON)
option(GTCLANG_BUILD_TESTING "Override BUILD_TESTING if part of a multi-project build." OFF)
option(GTCLANG_BUILD_EXAMPLES "Override BUILD_EXAMPLES if part of a multi-project build." OFF)

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

include(CTest)
include(GNUInstallDirs)

# TODO move back to a find_package
if(NOT DEFINED CCACHE_EXECUTABLE)
  find_program(CCACHE_EXECUTABLE
    NAMES ccache
    DOC "Path to ccache executable"
  )
  mark_as_advanced(CCACHE_EXECUTABLE)
endif()
if(CCACHE_EXECUTABLE)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE})
endif()

# TODO move to file
# Git version string
if(NOT DEFINED GIT_EXECUTABLE)
  find_program(GIT_EXECUTABLE
    NAMES git
    DOC "Path to git executable"
  )
  mark_as_advanced(GIT_EXECUTABLE)
endif()
if(GIT_EXECUTABLE)
  execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE git_hash_short OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT DEFINED GTCLANG_GIT_HASH OR NOT "${GTCLANG_GIT_HASH}" STREQUAL "${git_hash_short}")
  set(GTCLANG_GIT_HASH ${git_hash_short} CACHE INTERNAL "git hash (short) of current head")
endif()

set(compiler "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
set(GTCLANG_FULL_VERSION
    "${GTClang_VERSION}-${GTCLANG_GIT_HASH}-${CMAKE_SYSTEM_PROCESSOR}-${compiler}"
    CACHE STRING "Full version string of gtclang" FORCE)
#endTODO move to FILE

# Path for Find*.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
endif()

# TODO move to file
# CXX flags
include(CheckCXXCompilerFlag)
function(add_cxx_compiler_flag_if_supported flag var)
  string(FIND "${var}" "${flag}" flag_already_set)
  if(flag_already_set EQUAL -1)
    check_cxx_compiler_flag("${flag}" flag_supported)
    if(flag_supported)
      list(APPEND ${var} "${flag}")
      set(${var} "${${var}}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

add_cxx_compiler_flag_if_supported("-Wall" GTCLANG_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-Wextra" GTCLANG_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-pedantic" GTCLANG_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-Wno-unused-parameter" GTCLANG_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-Wno-sign-compare" GTCLANG_CXX_FLAGS) # Would like to remove
add_cxx_compiler_flag_if_supported("-Wl,--unresolved-symbols=report-all" GTCLANG_LINKER_FLAGS)

# Target properties
set(GTCLANG_TARGET_PROPERTIES CXX_EXTENSIONS OFF)

if (BUILD_SHARED_LIBS)
  # Needed because we use object libraries
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Add cxx standard, include directories, and properties
function(target_add_gtclang_standard_props target)
  target_include_directories(${target}
    PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  )
  target_compile_features(${target} PUBLIC cxx_std_17)
  target_compile_options(${target} PRIVATE ${GTCLANG_CXX_FLAGS})
  set_target_properties(${target} PROPERTIES ${GTCLANG_TARGET_PROPERTIES})
endfunction()

# Project dependencies
find_package(Clang 6.0.0 REQUIRED)

include(FetchContent)

# Do not attempt to update packages -- speeds up configure process
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Only test if BUILD_TESTING and main project, or GTCLANG_BUILD_TESTING is on
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR GTCLANG_BUILD_TESTING) AND BUILD_TESTING)
  set(${PROJECT_NAME}_TESTING ON)
else()
  set(${PROJECT_NAME}_TESTING OFF)
endif()
mark_as_advanced(${PROJECT_NAME}_TESTING)

if(NOT DAWN_GTCLANG_MULTIPROJECT)
  find_package(Dawn REQUIRED)
endif()

if(${PROJECT_NAME}_TESTING AND NOT TARGET gtest)
  # Google Test
  set(INSTALL_GTEST OFF)
  set(BUILD_GMOCK OFF)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.10.0
    )
  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  endif()
endif()

add_subdirectory(src)

if(${PROJECT_NAME}_TESTING)
  # Optionally enable CUDA for integration tests
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_ARCH "sm_60" CACHE STRING "CUDA architecture (e.g. sm_35, sm_60, sm_70)" )
  else()
    message(STATUS "Could NOT find CUDA compiler. Disabling GPU integration tests.")
  endif()

  cmake_dependent_option(GTCLANG_BUILD_TESTING_GT_MC "Build integration tests with the GridTools MC backend" ON "BUILD_TESTING" OFF)
  cmake_dependent_option(GTCLANG_BUILD_TESTING_GT_CUDA "Build integration tests with the GridTools CUDA backend" ON "BUILD_TESTING;CMAKE_CUDA_COMPILER" OFF)
  cmake_dependent_option(GTCLANG_BUILD_TESTING_PLAIN_CUDA "Build integration tests with the plain CUDA backend" ON "BUILD_TESTING;CMAKE_CUDA_COMPILER" OFF)

  add_subdirectory(test)
endif()
