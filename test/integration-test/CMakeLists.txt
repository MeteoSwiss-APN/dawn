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

include(mchbuildAddCustomDummyTarget)

set(cwd ${CMAKE_CURRENT_LIST_DIR})

set(directories 
  ${cwd}/Accesses
  ${cwd}/Diagnostics
  ${cwd}/PassStageSplitter
  ${cwd}/PassMultiStageSplitter
  ${cwd}/PassSetCaches
  ${cwd}/PassStageReordering
  ${cwd}/PassStageMerger
  ${cwd}/PassFieldVersioning
  ${cwd}/PassTemporaryMerger
  ${cwd}/PassTemporaryType
  ${cwd}/PassPreprocessor
  ${cwd}/PassSetNonTempCaches
  ${cwd}/SIR
)

list(APPEND compiler_flags -std=c++11 -O2)
list(APPEND library_flags ${Boost_LIBRARIES})

list(APPEND gridtools_flags ${compiler_flags} ${include_flags} ${library_flags})

gtclang_add_integrationtest(NAME no-codegen
                            GTCLANG ${GTCLANG_EXECUTABLE}
                            DIRECTORIES ${directories}
                            GRIDTOOLS_FLAGS ${gridtools_flags}
)

# Add a dummy target with all the source files so that they are included in a qt creator project
mchbuild_add_custom_dummy_target(NAME GTClangQtCreatorParseProjectIntegrationTest
                             DIRECTORIES ${directories})

# Codegen tests require GridTools
if(GTCLANG_HAS_GRIDTOOLS)
  # Add a dummy target with all the source files so that they are included in a qt creator project
  mchbuild_add_custom_dummy_target(NAME GTClangQtCreatorParseProjectCodeGenTest
                             DIRECTORIES ${cwd}/CodeGen)

  set(codegen_examples copy_stencil coriolis_stencil hori_diff_stencil_01 hori_diff_stencil_02 hori_diff_type2_stencil 
      hd_smagorinsky intervals_stencil globals_stencil stencil_functions nested_stencil_functions stencil_desc_ast)
  set(codegen_example_benchmarks)
  set(codegen_generate_targets)

  set(include_dirs)
  list(APPEND include_dirs "${CMAKE_SOURCE_DIR}/src")
  list(APPEND include_dirs "${GRIDTOOLS_INCLUDE_DIRS}")
  list(APPEND include_dirs "${CMAKE_SOURCE_DIR}")

  foreach(example ${codegen_examples})

    set(config_str "")
    if(EXISTS "${cwd}/CodeGen/${example}.json")
      set(config_str "--config=${cwd}/CodeGen/${example}.json")
    endif()

    file(MAKE_DIRECTORY "${cwd}/CodeGen/generated/")

    # Add make target 
    add_custom_command( OUTPUT ${cwd}/CodeGen/generated/${example}_gridtools.cpp
      COMMAND $<TARGET_FILE:gtclang> "-backend=gridtools" "${config_str}" "-o" "${cwd}/CodeGen/generated/${example}_gridtools.cpp"  "${cwd}/CodeGen/${example}.cpp"
      DEPENDS ${cwd}/CodeGen/${example}.cpp gtclang
    )

    add_custom_command(OUTPUT ${cwd}/CodeGen/generated/${example}_c++-naive.cpp
      COMMAND $<TARGET_FILE:gtclang> "-backend=c++-naive" "${config_str}" "-o" "${cwd}/CodeGen/generated/${example}_c++-naive.cpp"  "${cwd}/CodeGen/${example}.cpp"
      DEPENDS ${cwd}/CodeGen/${example}.cpp gtclang
    )

    list(APPEND ${example}_benchmarks ${cwd}/CodeGen/${example}_benchmark.cpp)
    list(APPEND codegen_generate_targets ${example}-codegen-gt)
    list(APPEND codegen_generate_targets ${example}-codegen-cxxnaive)

    add_executable(${example}_benchmarks "${cwd}/CodeGen/${example}_benchmark.cpp" "${cwd}/CodeGen/TestMain.cpp" "${cwd}/CodeGen/Options.cpp")
    target_include_directories(${example}_benchmarks PUBLIC ${include_dirs})
#TODO mchbuild_add_unittest does not work here since the DEPENDS argument not only makes a dependency but also links against them
    add_dependencies(${example}_benchmarks gtclang)
    target_link_libraries(${example}_benchmarks ${GTCLANG_UNITTEST_EXTERNAL_LIBRARIES} gtest)

    set_property(SOURCE ${cwd}/CodeGen/${example}_benchmark.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/CodeGen/generated/${example}_gridtools.cpp)
    set_property(SOURCE ${cwd}/CodeGen/${example}_benchmark.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/CodeGen/generated/${example}_c++-naive.cpp)

    add_test(NAME CTest-${example}_benchmarks COMMAND $<TARGET_FILE:${example}_benchmarks> 12 12 10 )
  endforeach()

endif()
