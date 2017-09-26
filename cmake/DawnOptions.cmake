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

# Building
set(BUILD_IS_NOT_RELEASE ON)
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  set(BUILD_IS_NOT_RELEASE OFF)
endif()
option(DAWN_ASSERTS "Enable asserts" ${BUILD_IS_NOT_RELEASE})
option(DAWN_USE_CCACHE "Use compile cache (ccache)" ON)

# Testing
option(DAWN_TESTING "Enable testing" ON)

# Documentation
option(DAWN_DOCUMENTATION "Enable documentation" OFF)
