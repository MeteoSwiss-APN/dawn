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
option(DAWN_BUILD_EXAMPLES "Override BUILD_EXAMPLES if part of a multi-project build." OFF)
set(PROTOBUF_PYTHON_MODULE_DIR "" CACHE PATH "Path to protobuf python module, if not at $Protobuf_INCLUDE_DIR/..")
mark_as_advanced(PROTOBUF_PYTHON_MODULE_DIR)

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

include(GNUInstallDirs)

# Path for Find*.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

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
endif("${isSystemDir}" STREQUAL "-1")

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

# Build up DAWN_CXX_FLAGS -- used to compile dawn sources later
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
function(target_add_dawn_standard_props target)
  target_include_directories(${target}
    PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  )
  target_compile_features(${target} PUBLIC cxx_std_17)
  target_compile_options(${target} PRIVATE ${DAWN_CXX_FLAGS})
  set_target_properties(${target} PROPERTIES ${DAWN_TARGET_PROPERTIES})
endfunction()

include(FetchContent)

# Do not attempt to update packages -- speeds up configure process
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Protobuf
find_package(Protobuf 3.4.0)
if(NOT Protobuf_FOUND)
  message(STATUS " Fetching Protobuf...")
  # Configuration options for Protobuf
  set(protobuf_BUILD_EXAMPLES OFF)
  set(protobuf_BUILD_TESTS OFF)
  set(protobuf_INSTALL_EXAMPLES OFF)
  # If not using bundle, we at least need protoc binary
  set(protobuf_BUILD_PROTOC_BINARIES ON)

  FetchContent_Declare(protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v3.10.1
  )

  FetchContent_GetProperties(protobuf)
  if(NOT protobuf_POPULATED)
    FetchContent_Populate(protobuf)
    add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR})
  endif()

  # Protobuf in build dir does not add standard headers
  set(PROTOBUF_ADD_ARGS -I${protobuf_SOURCE_DIR}/src)

  if(Python3_FOUND)
    add_custom_target(build-protobuf-python
      COMMAND PROTOC=$<TARGET_FILE:protobuf::libprotoc> ${Python3_EXECUTABLE} setup.py build
      WORKING_DIRECTORY ${protobuf_SOURCE_DIR}/python
    )
    add_custom_target(protobuf-python
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${protobuf_SOURCE_DIR}/python ${protobuf_BINARY_DIR}/python
      DEPENDS build-protobuf-python
    )
  endif()
endif()

# Only test if BUILD_TESTING and main project, or DAWN_BUILD_TESTING is on
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR DAWN_BUILD_TESTING) AND BUILD_TESTING)
  set(${PROJECT_NAME}_TESTING ON)
else()
  set(${PROJECT_NAME}_TESTING OFF)
endif()
mark_as_advanced(${PROJECT_NAME}_TESTING)
if(${PROJECT_NAME}_TESTING)
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
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  endif()

  enable_testing()
endif()

add_subdirectory(src)

find_package(Python3 COMPONENTS Interpreter)
if (Python3_FOUND)
  # Look for protobuf python module
  find_path(PROTOBUF_PYTHON_DIR google/protobuf/__init__.py
    PATHS ${Protobuf_INCLUDE_DIR}/../python ${PROTOBUF_PYTHON_MODULE_DIR}
    DOC "Path to protobuf python module")
  mark_as_advanced(PROTOBUF_PYTHON_DIR)
  if (PROTOBUF_PYTHON_DIR)
    set(ENABLE_PYTHON ON)
  endif()
  add_subdirectory(python)
else()
  message(STATUS "Python3 not found, not installing python bindings")
endif()

# Only build examples if BUILD_EXAMPLES and main project, or DAWN_BUILD_EXAMPLES is on
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR DAWN_BUILD_EXAMPLES) AND BUILD_EXAMPLES)
  set(${PROJECT_NAME}_EXAMPLES ON)
else()
  set(${PROJECT_NAME}_EXAMPLES OFF)
endif()
mark_as_advanced(${PROJECT_NAME}_EXAMPLES)

if(${PROJECT_NAME}_EXAMPLES)
  add_subdirectory(examples)
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

if(${PROJECT_NAME}_TESTING)
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/${PROJECT_NAME}Config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}ConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
  FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
