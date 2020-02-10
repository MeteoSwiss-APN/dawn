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
include(GoogleTest)

add_executable(${PROJECT_NAME}IntegrationTest
  TestMain.cpp
  TestIIRDeserializer.cpp
  GenerateInMemoryStencils.cpp
)
target_add_dawn_standard_props(${PROJECT_NAME}IntegrationTest)
target_link_libraries(${PROJECT_NAME}IntegrationTest Dawn DawnUnittest gtest)
set_target_properties(${PROJECT_NAME}IntegrationTest PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
gtest_discover_tests(${PROJECT_NAME}IntegrationTest TEST_PREFIX "Dawn::Integration::" DISCOVERY_TIMEOUT 30)

add_executable(DawnUpdateIIRReferences
  RegenerateIIRMain.cpp
  GenerateInMemoryStencils.cpp
)
target_link_libraries(DawnUpdateIIRReferences Dawn DawnUnittest gtest)
target_add_dawn_standard_props(DawnUpdateIIRReferences)
file(COPY reference_iir DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
