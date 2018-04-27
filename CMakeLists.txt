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
cmake_minimum_required(VERSION 3.3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(MCHBUILD_ROOT CACHE PATH "path to mchbuild package")

if("${MCHBUILD_ROOT}" STREQUAL "")
  message(FATAL_ERROR "MCHBUILD_ROOT not found! Try specifying it in the environment via -DMCHBUILD_ROOT=<>")
endif()
list(APPEND CMAKE_MODULE_PATH "${MCHBUILD_ROOT}/cmake")
include(mchbuildInit)
mchbuild_init()

include(DawnCMakeInit)
set(DAWN_ROOT ${CMAKE_CURRENT_LIST_DIR})

dawn_cmake_init()

include(mchbuildAddLibrary)
include(mchbuildAddTargetClangFormat)
include(mchbuildAddTargetCleanAll)
include(mchbuildCombineLibraries)
include(mchbuildConfigureFile)
include(mchbuildCreatePackageString)
include(DawnExportPackage)
include(mchbuildGetGitHeadRevision)
include(mchbuildMakeStringPair)
include(mchbuildReportResult)
include(mchbuildSetCXXStandard)
include(mchbuildEnableFullRPATH)

# Include the Dawn specific options, definitions and macros
include(DawnOptions)
include(DawnDefinitions)
include(DawnMacros)

# Set C++ standard
mchbuild_set_cxx_standard(c++11)

# Set C++ flags
dawn_set_cxx_flags()

# Add custom targets
mchbuild_add_target_clean_all(
  ${CMAKE_BINARY_DIR}/docs 
  ${CMAKE_BINARY_DIR}/sphinx
  ${CMAKE_BINARY_DIR}/src
  ${CMAKE_BINARY_DIR}/test
  ${CMAKE_BINARY_DIR}/bin
  ${CMAKE_BINARY_DIR}/python
  ${CMAKE_BINARY_DIR}/dawn-cmake
)

# Output summary of the configuration
macro(make_config_string FIRST SECOND)
  mchbuild_make_string_pair(${FIRST} ": ${SECOND}" 20 out)
  list(APPEND config_info ${out})
endmacro()
make_config_string("Dawn version" ${DAWN_FULL_VERSION_STR})
make_config_string("Platform" ${MCHBUILD_PLATFORM_STRING})
make_config_string("Architecture" ${MCHBUILD_ARCHITECTURE_STRING})
make_config_string("Compiler" ${MCHBUILD_COMPILER_STRING})
make_config_string("Build type" ${CMAKE_BUILD_TYPE})
make_config_string("Asserts" ${DAWN_ASSERTS})
make_config_string("Build shared " ${BUILD_SHARED_LIBS})
make_config_string("Install prefix" ${CMAKE_INSTALL_PREFIX})
mchbuild_report_result("Configuration summary" ${config_info})

# Include the packages 
if(DAWN_DOCUMENTATION)
  set(doc_packages Python3 Sphinx LaTeX Doxygen)
endif()

foreach(package bash Boost Protobuf ccache clang-format ${doc_packages})
  include("Add${package}")
  mchbuild_create_package_string(${package} info)
  list(APPEND package_info ${info})

  string(TOUPPER ${package} PACKAGE)
  list(APPEND DAWN_EXTERNAL_LIBRARIES ${DAWN_${PACKAGE}_LIBRARIES})
  list(APPEND DAWN_EXTERNAL_INCLUDE_DIRS ${DAWN_${PACKAGE}_INCLUDE_DIRS})
  list(APPEND DAWN_EXTERNAL_DEFINITIONS ${DAWN_${PACKAGE}_DEFINITIONS})

endforeach()

# Enable RPath support
mchbuild_enable_full_rpath("${CMAKE_INSTALL_PREFIX}/${DAWN_INSTALL_LIB_DIR};${DAWN_PROTOBUF_RPATH_DIR}")


include_directories(SYSTEM ${DAWN_EXTERNAL_INCLUDE_DIRS})
add_definitions(${DAWN_EXTERNAL_DEFINITIONS})

mchbuild_report_result("Package summary" ${package_info})

# Add clang-format target
mchbuild_add_target_clang_format(DIRECTORIES ${CMAKE_SOURCE_DIR}/src EXTENSION ".h;.cpp")

# Build dawn
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/src)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/test/utils/googletest/include)

add_subdirectory(src)
add_subdirectory(cmake)

if(DAWN_PYTHON)
  add_subdirectory(python)
endif()

if(DAWN_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(DAWN_DOCUMENTATION)
  add_subdirectory(docs)
endif()

# Install headers
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/src/ 
  DESTINATION ${DAWN_INSTALL_INCLUDE_DIR} 
  FILES_MATCHING PATTERN "*.h" PATTERN "*.inc" PATTERN "*.hpp"
)

# Install CMake package configuration
dawn_gen_install_config()
