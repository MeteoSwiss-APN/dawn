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

include(mchbuildExportOptions)
include(CMakeDependentOption)

set(BUILD_IS_NOT_RELEASE ON)
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  set(BUILD_IS_NOT_RELEASE OFF)
endif()

# Building
option(GTCLANG_ASSERTS "Enable asserts" ${BUILD_IS_NOT_RELEASE})
option(GTCLANG_USE_CCACHE "Use compile cache (ccache)" ON)
option(GTCLANG_BUILD_EXAMPLES "Build examples (requires boost, gridtools and OpenMP/CUDA)" ON)
option(GTCLANG_BUILD_EXAMPLES_WITH_GPU "Use the gpu (CUDA) backend when compiling gridtools" OFF)

# Testing
option(GTCLANG_TESTING "Enable testing" ON)
CMAKE_DEPENDENT_OPTION(GTCLANG_UNIT_TESTING 
                       "Enable unit testing" ON "GTCLANG_TESTING" OFF)
CMAKE_DEPENDENT_OPTION(GTCLANG_INTEGRATION_TESTING 
                       "Enable integration testing" ON "GTCLANG_TESTING" OFF)

# Documentation
option(GTCLANG_DOCUMENTATION "Enable documentation" OFF)

# Export options for meta projects
mchbuild_export_options(GTCLANG 
  GTCLANG_ASSERTS 
  GTCLANG_USE_CCACHE
  GTCLANG_BUILD_EXAMPLES
  GTCLANG_BUILD_EXAMPLES_WITH_GPU
  GTCLANG_TESTING
  GTCLANG_UNIT_TESTING
  GTCLANG_INTEGRATION_TESTING
  GTCLANG_DOCUMENTATION
)
