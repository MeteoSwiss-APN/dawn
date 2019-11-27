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

if(GTCLANG_FORCE_ATLAS_INTEGRATION_TESTS)
    find_package(eckit REQUIRED)
    find_package(atlas REQUIRED)
else()
    find_package(eckit)
    if(NOT eckit_FOUND)
            message(WARNING "Couldn't find eckit. Atlas integration tests are disabled.")
    endif()
    find_package(atlas)
    if(NOT atlas_FOUND)
            message(WARNING "Couldn't find atlas. Atlas integration tests are disabled.")
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
    yoda_add_executable(
      NAME AtlasIntegrationTestCodeGenerate
      SOURCES GenerateUnstructuredStencils.cpp
      DEPENDS Dawn::DawnShared
      OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/atlas-integrationtest
    )

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


    # Compare output of the generated stencils with output from reference implementation through calls to atlas_interface.hpp.
    yoda_add_unittest(
        NAME AtlasIntegrationTestCompareOutput
        SOURCES TestCompareOutputMain.cpp AtlasIntegrationTestCompareOutput.cpp AtlasVerifyer.h ${generated_stencil_codes_str} ${reference_stencil_codes_str}
        DEPENDS eckit atlas gtest
        OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/atlas-integrationtest
        GTEST_ARGS --gtest_color=yes --gtest_output=xml:AtlasIntegrationTestCompareOutput.xml
    )

    target_include_directories(AtlasIntegrationTestCompareOutput PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_include_directories(AtlasIntegrationTestCompareOutput PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/reference)
    target_include_directories(AtlasIntegrationTestCompareOutput PRIVATE ${CMAKE_SOURCE_DIR}/src)

endif()
