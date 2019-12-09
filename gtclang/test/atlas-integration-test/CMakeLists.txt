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

if(GTCLANG_REQUIRE_UNSTRUCTURED_TESTING)
    find_package(eckit REQUIRED)
    find_package(atlas REQUIRED)
else()
    find_package(eckit QUIET)
    if(NOT eckit_FOUND)
            message(STATUS "Couldn't find eckit. Atlas integration tests are disabled.")
    endif()
    find_package(atlas QUIET)
    if(NOT atlas_FOUND)
            message(STATUS "Couldn't find atlas. Atlas integration tests are disabled.")
    endif()
endif()

if(eckit_FOUND AND atlas_FOUND)

    # Need to specify here the names of the stencil codes that are going to be generated.
    set(generated_stencil_codes generated_copyCell.hpp
                                generated_copyEdge.hpp
                                generated_accumulateEdgeToCell.hpp
                                generated_verticalSum.hpp
                                generated_diffusion.hpp)
    set(reference_stencil_codes reference_diffusion.hpp)

    list(TRANSFORM generated_stencil_codes PREPEND generated/)
    list(TRANSFORM reference_stencil_codes PREPEND reference/)
    # NOTE: need to add the quotes here, otherwise CMAKE won't enforce the dependency on every file when setting OUTPUT of add_custom_command
    list(JOIN generated_stencil_codes " " generated_stencil_codes_str)
    list(JOIN reference_stencil_codes " " reference_stencil_codes_str)

    # Compile the executable that will programatically construct an IIR and code-generate it to file
    add_executable(AtlasIntegrationTestCodeGenerate GenerateUnstructuredStencils.cpp)
    target_link_libraries(AtlasIntegrationTestCodeGenerate Dawn DawnUnittest)

    # And run such executable. Outputs are the generated (stencil) codes: generated_*.hpp
    add_custom_target(AtlasIntegrationTestMkdirGenerated ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory generated)
    add_custom_command(
      OUTPUT ${generated_stencil_codes_str}
      COMMAND AtlasIntegrationTestCodeGenerate
      DEPENDS AtlasIntegrationTestMkdirGenerated
      COMMENT "Generating unstructured stencils to be tested"
    )

    # Copy references
    foreach(reference ${reference_stencil_codes})
        configure_file(${reference} ${reference} COPYONLY)
    endforeach()

    # TODO probably we want to put this skeleton in a add_gtclang_test
    set(test_name AtlasIntegrationTestCompareOutput)
    add_executable(${test_name}
      TestCompareOutputMain.cpp
      AtlasIntegrationTestCompareOutput.cpp
      AtlasVerifier.h
      ${generated_stencil_codes_str}
      ${reference_stencil_codes_str}
    )
    target_include_directories(AtlasIntegrationTestCompareOutput PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_include_directories(AtlasIntegrationTestCompareOutput PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/reference)
    target_include_directories(AtlasIntegrationTestCompareOutput PRIVATE ${CMAKE_SOURCE_DIR}/src)

    target_add_gtclang_standard_props(${test_name})
    target_link_libraries(${test_name} ${PROJECT_NAME} eckit atlas gtest)

    set_target_properties(${test_name} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/unittest
    )

    add_test(NAME ${test_name}
      COMMAND $<TARGET_FILE:${test_name}> --gtest_output=xml:${test_name}_unittest.xml
    )
endif()
