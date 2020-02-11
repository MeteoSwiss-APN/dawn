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

file(STRINGS "version.txt" __GTCLANG_VERSION)
string(REPLACE "." ";" __VERSION_LIST ${__GTCLANG_VERSION})

list(GET __VERSION_LIST 0 VERSION_MAJOR)
list(GET __VERSION_LIST 1 VERSION_MINOR)
list(GET __VERSION_LIST 2 VERSION_PATCH)
unset(__VERSION_LIST)

project(GTClang
  VERSION ${__GTCLANG_VERSION}
  HOMEPAGE_URL https://meteoswiss-apn.github.io/dawn/
  LANGUAGES CXX C
)
unset(__GTCLANG_VERSION)

# Useful standard options:
# - BUILD_TESTING<Bool>: Build test target (if only project, otherwise use GTCLANG_BUILD_TESTING to override).
# - BUILD_SHARED_LIBS<Bool>: Build gtclang shared libraries instead of static. If true, requires dependencies to be built as position independent code
# - CMAKE_PREFIX_PATH<String>: Semi-colon separated path to dependencies.

# Other options
include(CMakeDependentOption)
option(BUILD_EXAMPLES "Build examples" ON)
option(ENABLE_CUDA_IF_FOUND "Enable CUDA if it is detected" ON)
option(GTCLANG_BUILD_TESTING "Override BUILD_TESTING if part of a multi-project build." OFF)

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# Path for Find*.cmake files
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

include(CTest)
include(GNUInstallDirs)
include(EnableCCache)
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON) # Do not attempt to update packages -- speeds up configure process

include(GitVersionHash)
get_git_version_hash(__GTClang_GIT_HASH)
set(__compiler "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
set(GTCLANG_FULL_VERSION
    "${GTClang_VERSION}-${__GTClang_GIT_HASH}-${CMAKE_SYSTEM_PROCESSOR}-${__compiler}"
    CACHE INTERNAL "Full version string of gtclang")
unset(__GTClang_GIT_HASH)
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
endif()

include(AddCXXCompilerFlagIfSuppported)
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

# Only test if BUILD_TESTING and main project, or GTCLANG_BUILD_TESTING is on
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR GTCLANG_BUILD_TESTING) AND BUILD_TESTING)
  set(${PROJECT_NAME}_TESTING ON)
else()
  set(${PROJECT_NAME}_TESTING OFF)
endif()
mark_as_advanced(${PROJECT_NAME}_TESTING)

if(NOT TARGET Dawn::Dawn)
  find_package(Dawn REQUIRED)
endif()

if(${PROJECT_NAME}_TESTING AND NOT TARGET gtest)
  include(FetchGoogleTest)
endif()

add_subdirectory(src)

if(${PROJECT_NAME}_TESTING)
  # Optionally enable CUDA for integration tests
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_ARCH "sm_60" CACHE STRING "CUDA architecture (e.g. sm_35, sm_60, sm_70)")
  else()
    message(STATUS "Could NOT find CUDA compiler. Disabling GPU integration tests.")
  endif()

  cmake_dependent_option(GTCLANG_BUILD_TESTING_GT_MC
    "Build integration tests with the GridTools MC backend"
    ON "BUILD_TESTING" OFF
  )
  cmake_dependent_option(GTCLANG_BUILD_TESTING_GT_CUDA
    "Build integration tests with the GridTools CUDA backend"
    ON "BUILD_TESTING;CMAKE_CUDA_COMPILER;NOT ENABLE_CUDA_IF_FOUND" OFF
  )
  cmake_dependent_option(GTCLANG_BUILD_TESTING_PLAIN_CUDA
    "Build integration tests with the plain CUDA backend"
    ON "BUILD_TESTING;CMAKE_CUDA_COMPILER;NOT ENABLE_CUDA_IF_FOUND" OFF
  )

  add_subdirectory(test)
endif()

#TODO put this at the right place
#TODO: REMOVE THIS, it's a workaround to add driver-includes from dawn, otherwise parsing fails.
# set(GTCLANG_DSL_INCLUDES "${GTCLANG_DSL_INCLUDES}" "${DAWN_INCLUDE_DIRS}")

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
  FILES cmake/FindLLVM.cmake cmake/FindClang.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
  FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
  DIRECTORY ${DAWN_DRIVER_INCLUDEDIR}/driver-includes #TODO check location
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.inc" PATTERN "*.hpp"
)
