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

file(STRINGS "version.txt" __DAWN_VERSION)
string(REPLACE "." ";" __VERSION_LIST ${__DAWN_VERSION})

list(GET __VERSION_LIST 0 VERSION_MAJOR)
list(GET __VERSION_LIST 1 VERSION_MINOR)
list(GET __VERSION_LIST 2 VERSION_PATCH)
unset(__VERSION_LIST)

project(Dawn
  VERSION ${__DAWN_VERSION}
  HOMEPAGE_URL https://meteoswiss-apn.github.io/dawn/
  LANGUAGES CXX C
)
unset(__DAWN_VERSION)

# Useful standard options:
# - BUILD_TESTING<Bool>: Build test target (if only project, otherwise use DAWN_BUILD_TESTING to override).
# - BUILD_SHARED_LIBS<Bool>:
#     Build dawn as shared libraries instead of static. If true,
#     requires dependencies to be built as position independent code.
#     In any case, Dawn objects are built as position independent code to allow
#     building a shared library for python bindings.
# - CMAKE_PREFIX_PATH<String>: Semi-colon separated path to dependencies.

# Other options
option(BUILD_EXAMPLES "Build examples" ON)
option(DAWN_BUILD_TESTING "Override BUILD_TESTING if part of a multi-project build." OFF)
option(DAWN_BUILD_EXAMPLES "Override BUILD_EXAMPLES if part of a multi-project build." OFF)
option(DAWN_REQUIRE_PYTHON "Require enabling python." OFF)
set(PROTOBUF_PYTHON_MODULE_DIR "" CACHE PATH "Path to protobuf python module. Use this if CMake is not finding the module.")
set(DAWN4PY_MODULE_DIR "${PROJECT_SOURCE_DIR}/src" CACHE PATH "Path to dawn4py python module being built. Change this to self.build_lib for pip out-of-source builds.")
mark_as_advanced(PROTOBUF_PYTHON_MODULE_DIR DAWN4PY_MODULE_DIR)

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# Path for external CMake files
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CTest)
include(GNUInstallDirs)
include(EnableCCache)
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON) # Do not attempt to update packages -- speeds up configure process

include(GitVersionHash)
get_git_version_hash(__Dawn_GIT_HASH)
set(__compiler "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
set(DAWN_FULL_VERSION
    "${Dawn_VERSION}-${__Dawn_GIT_HASH}-${CMAKE_SYSTEM_PROCESSOR}-${__compiler}"
    CACHE INTERNAL "Full version string of dawn")
unset(__Dawn_GIT_HASH)
unset(__compiler)

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
endif("${isSystemDir}" STREQUAL "-1")

include(AddCXXCompilerFlagIfSuppported)
add_cxx_compiler_flag_if_supported("-Wall" DAWN_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-Wextra" DAWN_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-pedantic" DAWN_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-Wno-unused-parameter" DAWN_CXX_FLAGS)
add_cxx_compiler_flag_if_supported("-Wno-sign-compare" DAWN_CXX_FLAGS) # Would like to remove
add_cxx_compiler_flag_if_supported("-Wl,--unresolved-symbols=report-all" DAWN_LINKER_FLAGS)

# Target properties
set(DAWN_TARGET_PROPERTIES CXX_EXTENSIONS OFF)

# Position independent code is enabled globally
# - we need a shared library for python support (even if BUILD_SHARED_LIBS=OFF)
# - parts of our testing relies on python -> currently no strong reason for a no-python option
# - dependencies, e.g. protobuf, needs to be built with PIC as well
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter)

# Add cxx standard, include directories, and properties
function(target_add_dawn_standard_props target)
  target_include_directories(${target}
    PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_compile_features(${target} PUBLIC cxx_std_17)
  target_compile_options(${target} PRIVATE ${DAWN_CXX_FLAGS})
  set_target_properties(${target} PROPERTIES ${DAWN_TARGET_PROPERTIES})
endfunction()

# Protobuf
if(NOT DAWN_BUILD_PROTOBUF)
  find_package(Protobuf 3.4.0)
endif()
if(NOT Protobuf_FOUND)
  include(FetchProtobuf)
endif()

# Only test if BUILD_TESTING and main project, or DAWN_BUILD_TESTING is on
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR DAWN_BUILD_TESTING) AND BUILD_TESTING)
  set(${PROJECT_NAME}_TESTING ON)
else()
  set(${PROJECT_NAME}_TESTING OFF)
endif()
mark_as_advanced(${PROJECT_NAME}_TESTING)

if(${PROJECT_NAME}_TESTING)
  include(FetchGoogleTest)
endif()

if(Python3_FOUND)
  # TODO can we make this cleaner?
  if(DAWN_BUILD_PROTOBUF)
    # we know where it will be after the build step
    set(PROTOBUF_PYTHON_DIR ${protobuf_BINARY_DIR}/python)
  else()
    # we try to find it -- CMake has poor path detection
    find_path(PROTOBUF_PYTHON_DIR google/protobuf/__init__.py
      HINTS ${Protobuf_DIR}/../../.. ${PROTOBUF_PYTHON_MODULE_DIR}
      PATHS /usr/lib/python3/dist-packages
      PATH_SUFFIXES python python3 lib lib/python lib/python3 include/python include/python3
      DOC "Path to protobuf python module"
    )
  endif()
  mark_as_advanced(PROTOBUF_PYTHON_DIR)
  if(PROTOBUF_PYTHON_DIR)
    set(ENABLE_PYTHON ON)
  else()
    set(ENABLE_PYTHON OFF)
    message(STATUS "PROTOBUF_PYTHON_DIR not found, not installing python bindings")
  endif()
else()
  message(STATUS "Python3 not found, not installing python bindings")
endif()
if(NOT ENABLE_PYTHON AND DAWN_REQUIRE_PYTHON)
  message(FATAL_ERROR "Python was requested, but required dependencies are not found")
endif()

add_subdirectory(src)

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

if(${PROJECT_NAME}_TESTING)
  add_subdirectory(test)
endif()

set(DAWN_DRIVER_INCLUDEDIR ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE INTERNAL
  "Absolute path to driver includes in the source directory (used for super-builds) "
)
set(DAWN_INSTALL_DRIVER_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/${PROJECT_NAME}Config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  PATH_VARS DAWN_INSTALL_DRIVER_INCLUDEDIR
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
