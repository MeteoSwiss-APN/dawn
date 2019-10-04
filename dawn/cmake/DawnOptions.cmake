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

include(yodaExportOptions)

set(BUILD_IS_NOT_RELEASE ON)
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  set(BUILD_IS_NOT_RELEASE OFF)
endif()

# Building
option(DAWN_ASSERTS "Enable asserts" ${BUILD_IS_NOT_RELEASE})
option(DAWN_USE_CCACHE "Use compile cache (ccache)" ON)
option(DAWN_PYTHON "Build and install the Python module interface to HIR" ON)
option(DAWN_JAVA "Build and install the java interface to HIR" ON)

# Testing
option(DAWN_TESTING "Enable testing" ON)

# Documentation
option(DAWN_DOCUMENTATION "Enable documentation" OFF)

# Build examples
option(DAWN_EXAMPLES "Compile the examples" OFF)

# Export options for meta projects
yoda_export_options(DAWN 
  DAWN_ASSERTS 
  DAWN_USE_CCACHE
  DAWN_TESTING
  DAWN_DOCUMENTATION
)
