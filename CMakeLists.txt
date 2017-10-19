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
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries." FORCE)
endif()

if(NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE STRING
      "Install path prefix, prepended onto install directories." FORCE)
endif()

project(gtclang CXX)
cmake_minimum_required(VERSION 3.3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Locate Dawn for the CMake modules
include(DawnCMakeInit)
dawn_cmake_init()

include(DawnAddExecutable)
include(DawnAddLibrary)
include(DawnAddTargetCleanAll)
include(DawnAddTargetClangFormat)
include(DawnCheckInSourceBuild)
include(DawnCombineLibraries)
include(DawnConfigureFile)
include(DawnCreatePackageString)
include(DawnExportPackage)
include(DawnGetArchitectureInfo)
include(DawnGetCompilerInfo)
include(DawnGetGitHeadRevision)
include(DawnGetPlatformInfo)
include(DawnMakeStringPair)
include(DawnReportResult)
include(DawnSetCXXStandard)

dawn_get_compiler_info()
dawn_get_platform_info()
dawn_get_architecture_info()

# Include the GTClang specific options, definitions and macros
include(GTClangOptions)
include(GTClangDefinitions)
include(GTClangMacros)

# Set C++ standard
dawn_set_cxx_standard(c++11)

# Set C++ flags (Note that the LLVM/Clang package might add some other flags)
gtclang_set_cxx_flags()

# Prevent in source builds
dawn_check_in_source_build()

# Add custom targets
dawn_add_target_clean_all()

# Output summary of the configuration
macro(make_config_string FIRST SECOND)
  dawn_make_string_pair(${FIRST} ": ${SECOND}" 20 out)
  list(APPEND config_info ${out})
endmacro()
make_config_string("gtclang version" ${GTCLANG_FULL_VERSION})
make_config_string("Platform" ${DAWN_PLATFORM_STRING})
make_config_string("Architecture" ${DAWN_ARCHITECTURE_STRING})
make_config_string("Compiler" ${DAWN_COMPILER_STRING})
make_config_string("Build type" ${CMAKE_BUILD_TYPE})
make_config_string("Asserts" ${GTCLANG_ASSERTS})
dawn_report_result("Configuration summary" ${config_info})

# Include the packages (and set the correct, libraries, includes etc.)
foreach(package ccache Python3 Boost clang-format Clang Dawn GridTools OpenMP Threads)
  include("Add${package}")
  dawn_create_package_string(${package} info)
  list(APPEND package_info ${info})

  string(TOUPPER ${package} PACKAGE)
  list(APPEND GTCLANG_EXTERNAL_LIBRARIES ${DAWN_${PACKAGE}_LIBRARIES})
  list(APPEND GTCLANG_EXTERNAL_INCLUDE_DIRS ${DAWN_${PACKAGE}_INCLUDE_DIRS})
  list(APPEND GTCLANG_EXTERNAL_DEFINITIONS ${DAWN_${PACKAGE}_DEFINITIONS})
endforeach()

include_directories(SYSTEM ${GTCLANG_EXTERNAL_INCLUDE_DIRS})
add_definitions(${GTCLANG_EXTERNAL_DEFINITIONS})

# Output summary of the packages
dawn_report_result("Package summary" ${package_info})

# Add clang-format target
dawn_add_target_clang_format(DIRECTORIES ${CMAKE_SOURCE_DIR}/src EXTENSION ".h;.cpp")

# Build gtclang
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/src)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/test/utils/googletest/include)

add_subdirectory(src)

if(GTCLANG_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(GTCLANG_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(GTCLANG_DOCUMENTATION)
  add_subdirectory(docs)
endif()

# Install headers
install(
  DIRECTORY src/ 
  DESTINATION ${GTCLANG_INSTALL_INCLUDE_DIR} 
  FILES_MATCHING PATTERN "*.h" PATTERN "*.inc" PATTERN "*.hpp"
)
