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

if(eckit_DIR AND atlas_DIR)

    find_package(eckit)
    if(NOT eckit_FOUND)
            message(FATAL_ERROR "Could not find eckit")
    endif()

    find_package(atlas)
    if(NOT atlas_FOUND)
            message(FATAL_ERROR "Could not find atlas")
    endif()

    # Need to specify here the names of the stencil codes that are going to be generated.
    # NOTE: need to keep the quotes here, otherwise CMAKE won't enforce the dependency on every file when setting OUTPUT of add_custom_command
    set(generated_stencil_codes "generated_copyCell.hpp generated_copyEdge.hpp generated_accumulateEdgeToCell.hpp generated_diffusion.hpp")
    set(reference_stencil_codes "reference_diffusion.hpp")


    # Compile the executable that will programatically construct an IIR and code-generate it to file
    yoda_add_executable(
      NAME AtlasIntegrationTestCodeGenerate
      SOURCES GenerateUnstructuredStencils.cpp
      DEPENDS ${GTCLANG_EXTERNAL_LIBRARIES} Dawn::DawnShared
      OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/atlas-integrationtest-codegenerate
    )
    # And run such executable. Outputs are the generated (stencil) codes: generated_*.hpp
    add_custom_command(
      OUTPUT ${generated_stencil_codes}
      COMMAND AtlasIntegrationTestCodeGenerate
      COMMENT "Generating unstructured stencils to be tested"
    )
    # Copy references
    foreach(reference ${reference_stencil_codes})
        configure_file(reference/${reference} ${reference} COPYONLY)
    endforeach()
    # Compare output of the generated stencils with output from reference implementation through calls to atlas_interface.hpp.
    yoda_add_unittest(
        NAME AtlasIntegrationTestCompareOutput
        SOURCES ${generated_stencil_codes} ${reference_stencil_codes} TestCompareOutputMain.cpp AtlasIntegrationTestCompareOutput.cpp
        DEPENDS eckit atlas gtest GTClangUnittestStatic
        OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/atlas-integrationtest-compareoutput
        GTEST_ARGS --gtest_color=yes --gtest_output=xml:AtlasIntegrationTestCompareOutput.xml
    )

    target_include_directories(AtlasIntegrationTestCompareOutput PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_include_directories(AtlasIntegrationTestCompareOutput PRIVATE ${CMAKE_SOURCE_DIR}/src)


else()
    message(STATUS "eckit_DIR and/or atlas_DIR not set, disabling atlas integration tests")
endif()
