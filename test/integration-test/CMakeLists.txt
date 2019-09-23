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

dawn_add_integrationtest_impl(
  NAME DawnIntegrationTest
  SOURCES TestMain.cpp
          TestIIRDeserializer.cpp
)

file(COPY reference_iir DESTINATION ${CMAKE_BINARY_DIR}/bin/integrationtest)
file(COPY reference_iir DESTINATION ${CMAKE_BINARY_DIR}/test/integration-test)