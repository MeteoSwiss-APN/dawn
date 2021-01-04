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

if (CMAKE_BUILD_TYPE STREQUAL "Debug")

# Need to specify here the names of the stencil codes that are going to be generated.
set(generated_stencil_codes generated_accumulateEdgeToCell.hpp
  generated_copyCell.hpp
  generated_copyEdge.hpp
  generated_diamond.hpp
  generated_diamondWeights.hpp
  generated_diffusion.hpp
  generated_globalVar.hpp
  generated_gradient.hpp
  generated_horizontalVertical.hpp
  generated_intp.hpp
  generated_iterationSpaceUnstructured.hpp
  generated_nestedSimple.hpp
  generated_nestedWithField.hpp
  generated_nestedWithSparse.hpp
  generated_padding.hpp
  generated_reductionAndFillWithCenterSparse.hpp
  generated_reductionInIfConditional.hpp
  generated_reductionWithCenter.hpp
  generated_reductionWithCenterSparse.hpp
  generated_sparseAssignment0.hpp  
  generated_sparseAssignment1.hpp 
  generated_sparseAssignment2.hpp  
  generated_sparseAssignment3.hpp 
  generated_sparseAssignment4.hpp 
  generated_sparseAssignment5.hpp
  generated_sparseDimension.hpp
  generated_sparseDimensionTwice.hpp
  generated_sparseTempFieldAllocation.hpp
  generated_tempFieldAllocation.hpp
  generated_tempField.hpp
  generated_tridiagonalSolve.hpp
  generated_verticalIndirecion.hpp
  generated_verticalSum.hpp
)

set(reference_stencil_codes reference_diffusion.hpp
  reference_gradient.hpp reference_intp.hpp reference_diamond.hpp reference_diamondWeights.hpp
)

list(TRANSFORM generated_stencil_codes PREPEND generated/)
list(TRANSFORM reference_stencil_codes PREPEND reference/)

# Compile the executable that will programatically construct an IIR and code-generate it to file
add_executable(AtlasIntegrationTestCodeGenerate GenerateUnstructuredStencils.cpp testMutator.cpp)

target_link_libraries(AtlasIntegrationTestCodeGenerate Dawn DawnUnittest)

add_custom_command(
  OUTPUT ${generated_stencil_codes}
  COMMAND ${CMAKE_COMMAND} -E make_directory generated
  COMMAND AtlasIntegrationTestCodeGenerate
  DEPENDS AtlasIntegrationTestCodeGenerate
  COMMENT "Generating unstructured stencils to be tested"
)

# Copy references
foreach(reference ${reference_stencil_codes})
  configure_file(${reference} ${reference} COPYONLY)
endforeach()

add_custom_target(AtlasIntegrationTestCopyReferences ALL SOURCES ${reference_stencil_codes})

# Atlas Tests
set(test_name AtlasIntegrationTestCompareOutput)
add_executable(${test_name}
  AtlasIntegrationTestCompareOutput.cpp
  UnstructuredVerifier.h
  AtlasCartesianWrapper.h
  AtlasCartesianWrapper.cpp
  ${generated_stencil_codes}
  ${reference_stencil_codes}
)

add_dependencies(${test_name} AtlasIntegrationTestCopyReferences)

target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/reference)
target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_add_dawn_standard_props(${test_name})
target_link_libraries(${test_name} ${PROJECT_NAME} eckit atlas gtest gtest_main)

set_target_properties(${test_name} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/unittest
)

add_test(NAME ${test_name}
  COMMAND $<TARGET_FILE:${test_name}> --gtest_output=xml:${test_name}_unittest.xml
)
# Toylib Tests
set(test_name ToylibIntegrationTestCompareOutput)
add_executable(${test_name}
  ToylibIntegrationTestCompareOutput.cpp
  UnstructuredVerifier.h
  ${generated_stencil_codes}
  ${reference_stencil_codes}
)

add_dependencies(${test_name} AtlasIntegrationTestCopyReferences)

target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/reference)
target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_add_dawn_standard_props(${test_name})
target_link_libraries(${test_name} ${PROJECT_NAME} toylib eckit atlas gtest gtest_main) #need atlas here unfortunately because of the verifier

set_target_properties(${test_name} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/unittest
)

gtest_discover_tests(${test_name} TEST_PREFIX "Dawn::Integration::Unstructured::"
  DISCOVERY_TIMEOUT 30
)

endif()
