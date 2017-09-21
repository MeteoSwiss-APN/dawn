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
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING 
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

project(gtclang C CXX)
cmake_minimum_required(VERSION 3.3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/modules")

##===-------------------------------------- CMake modules -------------------------------------===##
include(GTClangIncludeGuard)
include(GTClangOptions)
include(GTClangDefinitions)
include(GTClangMacros)
include(GTClangFunctions)

# Prevent in source builds
gtclang_check_in_source_build()

# Check compliance of compiler
gtclang_check_compiler()

# Custom tragets
gtclang_add_target_clean_all()
gtclang_add_target_clang_format(${CMAKE_SOURCE_DIR}/src)
gtclang_make_build_script()

# Output findings
gtclang_report_result(
  "Configuration summary" 
  "gridtools clang version: ${GTCLANG_VERSION_STRING}"
  "Platform: ${GTCLANG_PLATFORM_STRING}"
  "Compiler: ${GTCLANG_COMPILER_STRING}"
  "Build type: ${CMAKE_BUILD_TYPE}"
)

##===---------------------------------- Add external libraries --------------------------------===##
include(AddThreads)
include(AddClang)
include(AddGSL)

if(GTCLANG_TESTING)
  enable_testing()

  if(GTCLANG_INTEGRATION_TESTING)
    include(AddPython3)
    include(AddGTClang)
    include(AddGridTools)
    include(AddBoost)

    if(GTCLANG_GRIDTOOLS_FOUND AND GTCLANG_BOOST_FOUND)
      set(GTCLANG_HAS_GRIDTOOLS 1)
    endif()
  endif()
endif()

if(GTCLANG_BUILD_EXAMPLES)
  include(AddGTClang)
  include(AddGridTools)
  include(AddBoost)

  if(GTCLANG_BUILD_EXAMPLES_WITH_GPU)
    include(AddCUDA)
  else()
    include(AddOpenMP)
  endif()

  if(NOT(GTCLANG_GRIDTOOLS_FOUND) AND NOT(GTCLANG_BOOST_FOUND))
    message(STATUS "GirdTools and Boost are required to build examples")
    set(GTCLANG_BUILD_EXAMPLES OFF)
    set(GTCLANG_BUILD_EXAMPLES_WITH_GPU OFF)
  endif()
endif()

# Set C++ flags (including LLVM specific flags)
gtclang_set_cxx_standard(c++11)
gtclang_set_cxx_flags()

# Output findings
gtclang_report_result(
  "External libraries summary" 
  "${GTCLANG_THREADS_INFO_STR}"
  "${GTCLANG_CLANG_INFO_STR}"
  "${GTCLANG_GSL_INFO_STR}"
  "${GTCLANG_GRIDTOOLS_INFO_STR}"
  "${GTCLANG_BOOST_INFO_STR}"
  "${GTCLANG_PYTHON3_INFO_STR}"
  "${GTCLANG_GTCLANG_INFO_STR}"
  "${GTCLANG_OPENMP_INFO_STR}"
  "${GTCLANG_CUDA_INFO_STR}"
)

##===---------------------------------- Build gridtools clang ---------------------------------===##
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/test/utils/googletest/include)

add_subdirectory(src)

# if(GTCLANG_BUILD_EXAMPLES)
#   add_subdirectory(examples)
# endif()

##===---------------------------------- Testing -----------------------------------------------===##
if(GTCLANG_TESTING)
  add_subdirectory(test)
endif()

##===---------------------------------- Documentation -----------------------------------------===##
# if(GTCLANG_DOCUMENTATION)
#   add_subdirectory(docs)
# endif()

