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


add_executable(DawnIntegrationTest
  TestMain.cpp
  TestIIRDeserializer.cpp
  GenerateInMemoryStencils.cpp
)
target_link_libraries(DawnIntegrationTest Dawn DawnUnittest gtest)
target_compile_options(DawnIntegrationTest ${DAWN_CXX_FLAGS})
set_target_properties(DawnIntegrationTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME DawnIntegrationTest
  COMMAND $<TARGET_FILE:DawnIntegrationTest>
)

add_executable(DawnUpdateIIRReferences
  RegenerateIIRMain.cpp
  GenerateInMemoryStencils.cpp
)
target_link_libraries(DawnUpdateIIRReferences Dawn gtest)
target_compile_options(DawnUpdateIIRReferences ${DAWN_CXX_FLAGS})

file(COPY reference_iir DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/integrationtest
# file(COPY reference_iir DESTINATION ${CMAKE_BINARY_DIR}/bin/integrationtest)
