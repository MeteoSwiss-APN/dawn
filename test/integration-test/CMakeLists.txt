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

macro(ADD_COMPILER_FLAG_TO_EXAMPLE _KEY _VALUE)
  set("MyMap_${_KEY}" "${MyMap_${_KEY}}" "${_VALUE}")
endmacro(ADD_COMPILER_FLAG_TO_EXAMPLE)

macro(GET_COMPILER_FLAGS _KEY _RETVAL)
    if("${MyMap_${_KEY}}" STREQUAL "")
        set(${_RETVAL} "")
    else()
        set(${_RETVAL} ${MyMap_${_KEY}})
    endif()
endmacro(GET_COMPILER_FLAGS)

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
  ${cwd}/PassSetBoundaryCondition
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
# Add a dummy target with all the source files so that they are included in a qt creator project
mchbuild_add_custom_dummy_target(NAME GTClangQtCreatorParseProjectCodeGenTest
                         DIRECTORIES ${cwd}/CodeGen)

# Codegen tests require GridTools
if(GTCLANG_HAS_GRIDTOOLS AND GTCLANG_BUILD_EXAMPLES)
  set(codegen_examples copy_stencil coriolis_stencil hori_diff_stencil_01 hori_diff_stencil_02
      hori_diff_type2_stencil hd_smagorinsky intervals_stencil globals_stencil stencil_functions
      nested_stencil_functions stencil_desc_ast boundary_condition boundary_condition_2)
  set(codegen_example_benchmarks)
  set(codegen_generate_targets)

ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition_2" "-max-fields=2")
ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition_2" "-fsplit-stencils")

ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition" "-max-fields=2")
ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition" "-fsplit-stencils")
set(include_dirs)
list(APPEND include_dirs "${CMAKE_SOURCE_DIR}/src")
list(APPEND include_dirs "${GRIDTOOLS_INCLUDE_DIRS}")
list(APPEND include_dirs "${CMAKE_SOURCE_DIR}")
list(APPEND include_dirs "${Boost_INCLUDE_DIR}")

foreach(example ${codegen_examples})
    # Add json inputfiles if they exist
    set(config_str "")
    if(EXISTS "${cwd}/CodeGen/${example}.json")
      set(config_str "--config=${cwd}/CodeGen/${example}.json")
    endif()

    # Add the specified compilerflags
    GET_COMPILER_FLAGS(${example} flags)
    set(config_str "${config_str}" "${flags}")

    file(MAKE_DIRECTORY "${cwd}/CodeGen/generated/")

    # Add make target
    add_custom_command( OUTPUT ${cwd}/CodeGen/generated/${example}_gridtools.cpp
      COMMAND $<TARGET_FILE:gtclang> "-backend=gridtools" ${config_str} "-o" "${cwd}/CodeGen/generated/${example}_gridtools.cpp"  "${cwd}/CodeGen/${example}.cpp"
      DEPENDS ${cwd}/CodeGen/${example}.cpp gtclang
    )

    add_custom_command(OUTPUT ${cwd}/CodeGen/generated/${example}_c++-naive.cpp
      COMMAND $<TARGET_FILE:gtclang> "-backend=c++-naive" ${config_str} "-o" "${cwd}/CodeGen/generated/${example}_c++-naive.cpp"  "${cwd}/CodeGen/${example}.cpp"
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

    add_test(NAME CTest-${example}_benchmarks COMMAND $<TARGET_FILE:${example}_benchmarks> 12 12 10 --gtest_output=xml:${example}_unittest.xml)

    if(GTCLANG_BUILD_EXAMPLES_WITH_GPU)
      cuda_add_executable(${example}_benchmarks_cuda "${cwd}/CodeGen/${example}_benchmark.cu" "${cwd}/CodeGen/TestMain.cpp" "${cwd}/CodeGen/Options.cpp")

      target_include_directories(${example}_benchmarks_cuda PUBLIC ${include_dirs})
      add_dependencies(${example}_benchmarks_cuda gtclang ${example}_benchmarks)
      target_link_libraries(${example}_benchmarks_cuda gtest ${CUDA_CUDART_LIBRARY})
      target_compile_definitions(${example}_benchmarks_cuda PUBLIC GRIDTOOLS_CLANG_CUDA ${GTCLANG_DEFINITIONS})

      set_property(SOURCE ${cwd}/CodeGen/${example}_benchmark.cu APPEND PROPERTY OBJECT_DEPENDS ${cwd}/CodeGen/generated/${example}_gridtools.cpp)
      set_property(SOURCE ${cwd}/CodeGen/${example}_benchmark.cu APPEND PROPERTY OBJECT_DEPENDS ${cwd}/CodeGen/generated/${example}_c++-naive.cpp)

      if(CTEST_CUDA_SUBMIT) 
        set(CTEST_SUBMIT_COMMAND "${SLURM_SRUN_COMMAND}" "-n" "${GTCLANG_SLURM_N_TASKS}" "${GTCLANG_SLURM_RESOURCES}" "-p" "${GTCLANG_SLURM_PARTITION}")
      endif(CTEST_CUDA_SUBMIT)

      add_test(NAME CTest-${example}_benchmarks_cuda COMMAND ${CTEST_SUBMIT_COMMAND} $<TARGET_FILE:${example}_benchmarks_cuda> 12 12 10 --gtest_output=xml:${example}_cuda_unittest.xml)
    endif(GTCLANG_BUILD_EXAMPLES_WITH_GPU)

endforeach()

gtclang_add_codegentest(NAME codegen-only
                        GTCLANG ${GTCLANG_EXECUTABLE}
                        CODEGEN_EXAMPLES ${codegen_examples}
                        GRIDTOOLS_FLAGS ${gridtools_flags}
)

endif()
