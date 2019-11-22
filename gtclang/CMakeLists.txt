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

include(CheckLanguage)
check_language(CUDA)

# Useful standard options:
# - BUILD_TESTING<Bool>: Build test target (if only project, otherwise use DAWN_BUILD_TESTING to override).
# - BUILD_SHARED_LIBS<Bool>: Build dawn as shared libraries instead of static. If true, requires dependencies to be built as position independent code
# - CMAKE_PREFIX_PATH<String>: Semi-colon separated path to dependencies.

# Other options
option(BUILD_EXAMPLES "Build examples" ON)
option(GTCLANG_BUILD_TESTING "Override BUILD_TESTING if part of a multi-project build." OFF)

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# Path for Find*.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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

add_cxx_compiler_flag_if_supported("-Wall" DAWN_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-Wextra" DAWN_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-pedantic" DAWN_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-Wno-unused-parameter" DAWN_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-Wno-sign-compare" DAWN_CXX_FLAGS) # Would like to remove
add_cxx_compiler_flag_if_supported("-Wl,--unresolved-symbols=report-all" DAWN_LINKER_FLAGS)

# Target properties
set(DAWN_TARGET_PROPERTIES CXX_EXTENSIONS OFF)

# Needed because we use object libraries
if (BUILD_SHARED_LIBS)
  list(APPEND DAWN_TARGET_PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Add cxx standard, include directories, and properties
function(target_add_gtclang_standard_props target)
  target_include_directories(${target}
    PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  )
  target_compile_features(${target} PUBLIC cxx_std_17)
  target_compile_options(${target} PRIVATE ${DAWN_CXX_FLAGS})
  set_target_properties(${target} PROPERTIES ${DAWN_TARGET_PROPERTIES})
endfunction()

include(GNUInstallDirs)

# Project dependencies
# find_package(Python3 COMPONENTS Interpreter)
if(NOT DAWN_GTCLANG_MULTIPROJECT)
  find_package(Dawn REQUIRED)
endif()
find_package(LLVM REQUIRED)
find_package(Clang REQUIRED)
find_package(GridTools)
if(NOT GridTools)
  message(STATUS "GridTools not found. Disabling backend")
  set(BUILD_TESTING OFF)
endif()

add_subdirectory(src)

# if(GTCLANG_BUILD_EXAMPLES)
#   add_subdirectory(examples)
# endif()

# if(GTCLANG_TESTING)
#   enable_testing()
#   add_subdirectory(test)
# endif()

# if(GTCLANG_DOCUMENTATION)
#   add_subdirectory(docs)
# endif()
# find_package(clang)

# # Include the packages (and set the correct, libraries, includes etc.)
# foreach(package bash ccache Python3 Boost clang-format Clang Dawn OpenMP Threads)
#   include("Add${package}")
#   yoda_create_package_string(${package} info)
#   list(APPEND package_info ${info})

#   string(TOUPPER ${package} PACKAGE)
#   if(${PACKAGE}_FOUND)
#     list(APPEND GTCLANG_EXTERNAL_LIBRARIES ${YODA_${PACKAGE}_LIBRARIES})
#     list(APPEND GTCLANG_EXTERNAL_INCLUDE_DIRS ${YODA_${PACKAGE}_INCLUDE_DIRS})
#     list(APPEND GTCLANG_EXTERNAL_DEFINITIONS ${YODA_${PACKAGE}_DEFINITIONS})
#   endif()
# endforeach()

# if(GTCLANG_BUILD_GT_CPU_EXAMPLES OR GTCLANG_BUILD_GT_GPU_EXAMPLES OR GTCLANG_BUILD_CUDA_EXAMPLES)
#   include("AddGridTools")
#   if(GridTools_FOUND)
#     yoda_create_package_string(GridTools info)
#     list(APPEND package_info ${info})

#     string(TOUPPER GridTools PACKAGE)
#     list(APPEND GTCLANG_EXTERNAL_LIBRARIES ${YODA_${PACKAGE}_LIBRARIES})
#     list(APPEND GTCLANG_EXTERNAL_INCLUDE_DIRS ${YODA_${PACKAGE}_INCLUDE_DIRS})
#     list(APPEND GTCLANG_EXTERNAL_DEFINITIONS ${YODA_${PACKAGE}_DEFINITIONS})
#   endif()
# endif()

# include_directories(SYSTEM ${GTCLANG_EXTERNAL_INCLUDE_DIRS})
# add_definitions(${GTCLANG_EXTERNAL_DEFINITIONS})

# # Output summary of the packages
# yoda_report_result("Package summary" ${package_info})

# # Support of RPATH-exports of dawn
# if(NOT(DEFINED DAWN_RPATH_DIR))
#    message(FATAL_ERROR "DAWN_RPATH_DIR is not exported by DAWN")
# endif()
# yoda_enable_full_rpath("${DAWN_RPATH_DIR}")

# # Add clang-format target
# set(format_directories
#   "${CMAKE_SOURCE_DIR}/src"
#   "${CMAKE_SOURCE_DIR}/test/unit-test"
# )

# yoda_add_target_clang_format(DIRECTORIES ${format_directories} EXTENSION ".h;.cpp")

# # Build gtclang
# include_directories(${CMAKE_SOURCE_DIR}/src)
# include_directories(${CMAKE_BINARY_DIR}/src)
# include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/test/utils/googletest/include)


# gtclang_gen_install_config()

# # Install headers
# install(
#   DIRECTORY src/
#   DESTINATION ${GTCLANG_INSTALL_INCLUDE_DIR}
#   FILES_MATCHING PATTERN "*.h" PATTERN "*.inc" PATTERN "*.hpp"
# )
