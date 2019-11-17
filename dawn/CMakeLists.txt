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


cmake_minimum_required(VERSION 3.13)

# project version
set(VERSION_MAJOR 0)
set(VERSION_MINOR 0)
set(VERSION_PATCH 1)

set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

project(Dawn
  VERSION ${VERSION}
  HOMEPAGE_URL https://meteoswiss-apn.github.io/dawn/
  LANGUAGES CXX C)

# Useful standard options:
# - BUILD_TESTING<Bool>: Build test target (if only project, otherwise use DAWN_BUILD_TESTING to override).
# - BUILD_SHARED_LIBS<Bool>: Build dawn as shared libraries instead of static. If true, requires dependencies to be built as position independent code
# - CMAKE_PREFIX_PATH<String>: Semi-colon separated path to dependencies.

# Other options
option(BUILD_EXAMPLES "Build examples" ON)
option(DAWN_BUILD_TESTING "Override BUILD_TESTING if part of a multi-project build." OFF)

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
add_cxx_compiler_flag_if_supported("-Wno-sign-compare" DAWN_CXX_FLAGS) # TODO: Remove this!

add_cxx_compiler_flag_if_supported("-Wl,--unresolved-symbols=report-all" DAWN_LINKER_FLAGS)

if(BUILD_SHARED_LIBS)
add_cxx_compiler_flag_if_supported("-fPIC" DAWN_CXX_FLAGS)
endif()

# Target properties
set(DAWN_TARGET_PROPERTIES CXX_EXTENSIONS OFF)

if (BUILD_SHARED_LIBS)
  list(APPEND DAWN_TARGET_PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

include(GNUInstallDirs)

# Project dependencies
find_package(Protobuf 3.4.0 REQUIRED)
find_package(Python3 COMPONENTS Interpreter)

add_subdirectory(src)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if (Python3_FOUND)
  set(ENABLE_PYTHON ON)
  add_subdirectory(python)
else()
  message(STATUS "Python3 not found, not installing python bindings")
endif()

# Docs only available if this is the main app
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  find_package(Doxygen)
  find_package(Sphinx)
  if(Doxygen_FOUND AND SPHINX_FOUND)
	  add_subdirectory(docs)
  else()
	  message(STATUS "Doxygen and Sphinx not found, not building docs")
  endif()
endif()

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR DAWN_BUILD_TESTING) AND BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_NAME}ConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(
  FILES
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)
