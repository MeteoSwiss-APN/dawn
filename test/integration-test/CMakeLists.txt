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

include(yodaAddCustomDummyTarget)

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

function(code_generate_examples)
  set(oneValueArgs BACKEND)
  set(multiValueArgs EXAMPLES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  set(codegen_examples_ ${ARG_EXAMPLES})
  set(backend ${ARG_BACKEND})

  foreach(example ${codegen_examples_})
      # Add json inputfiles if they exist
      set(config_str "")
      if(EXISTS "${cwd}/CodeGen/${example}.json")
        set(config_str "--config=${cwd}/CodeGen/${example}.json")
      endif()
  
      if(CMAKE_BUILD_TYPE MATCHES DEBUG)
        set(config_str "${config_str}" "-fpass-verbose")
      endif(CMAKE_BUILD_TYPE MATCHES DEBUG)


      # Add the specified compilerflags
      GET_COMPILER_FLAGS(${example} flags)
      set(config_str "${config_str}" "${flags}")
  
      file(MAKE_DIRECTORY "${cwd}/CodeGen/generated/")
  
      # Add make target
      add_custom_command( OUTPUT ${cwd}/CodeGen/generated/${example}_${backend}.cpp
        COMMAND $<TARGET_FILE:gtclang> "-backend=${backend}" ${config_str} "-o" "${cwd}/CodeGen/generated/${example}_${backend}.cpp"  "${cwd}/CodeGen/${example}.cpp"
        DEPENDS ${cwd}/CodeGen/${example}.cpp gtclang
      )
 
      add_custom_target(${example}_${backend}_codegen ALL DEPENDS ${cwd}/CodeGen/generated/${example}_${backend}.cpp)
      set_source_files_properties(${cwd}/CodeGen/generated/${example}_${backend}.cpp PROPERTIES GENERATED TRUE)
 
      add_custom_command(OUTPUT ${cwd}/CodeGen/generated/${example}_c++-naive.cpp
        COMMAND $<TARGET_FILE:gtclang> "-backend=c++-naive" ${config_str} "-o" "${cwd}/CodeGen/generated/${example}_c++-naive.cpp"  "${cwd}/CodeGen/${example}.cpp"
        DEPENDS ${cwd}/CodeGen/${example}.cpp gtclang
      )
 
      add_custom_target(${example}_c++-naive_codegen ALL DEPENDS ${cwd}/CodeGen/generated/${example}_c++-naive.cpp)
      set_source_files_properties(${cwd}/CodeGen/generated/${example}_c++-naive.cpp PROPERTIES GENERATED TRUE)

  endforeach()
endfunction(code_generate_examples)

function(compile_generated_examples)
  set(oneValueArgs BACKEND)
  set(multiValueArgs EXAMPLES DUMMY)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  set(codegen_examples_ ${ARG_EXAMPLES})
  set(backend ${ARG_BACKEND})

  if(NOT ${backend} STREQUAL "gt") 
    message(FATAL "backend ${backend} not supported")
  endif()

  foreach(example ${ARG_DUMMY})
      add_executable(${example}_benchmarks_${backend} ${cwd}/CodeGen/dummy.cpp)
      set_property(SOURCE ${cwd}/CodeGen/dummy.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/CodeGen/generated/${example}_c++-naive.cpp)
      if(GTCLANG_BUILD_CUDA_EXAMPLES)
        set_property(SOURCE ${cwd}/CodeGen/dummy.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/CodeGen/generated/${example}_cuda.cpp)
      endif(GTCLANG_BUILD_CUDA_EXAMPLES)
  endforeach()
  foreach(example ${codegen_examples_})
      add_executable(${example}_benchmarks_${backend} "${cwd}/CodeGen/${example}_benchmark.cpp" "${cwd}/CodeGen/TestMain.cpp" "${cwd}/CodeGen/Options.cpp")
      target_include_directories(${example}_benchmarks_${backend} PUBLIC ${include_dirs})
      target_compile_definitions(${example}_benchmarks_${backend} PRIVATE -DOPTBACKEND=${backend})
      if(${backend} STREQUAL gt)
        target_compile_definitions(${example}_benchmarks_${backend} PRIVATE -DBACKEND_MC)
      endif()
      add_dependencies(${example}_benchmarks_${backend} gtclang ${example}_c++-naive_codegen ${example}_${backend}_codegen)
      target_link_libraries(${example}_benchmarks_${backend} ${GTCLANG_UNITTEST_EXTERNAL_LIBRARIES} gtest)

#      set_property(SOURCE ${cwd}/CodeGen/${example}_benchmark.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/CodeGen/generated/${example}_${backend}.cpp)
#      set_property(SOURCE ${cwd}/CodeGen/${example}_benchmark.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/CodeGen/generated/${example}_c++-naive.cpp)
  
      add_test(NAME CTest-${example}_benchmarks_${backend} COMMAND $<TARGET_FILE:${example}_benchmarks_${backend}> 12 12 10 --gtest_output=xml:${example}_unittest.xml)
  
  endforeach()
endfunction(compile_generated_examples)

function(cuda_compile_generated_examples)
  set(oneValueArgs BACKEND)
  set(multiValueArgs EXAMPLES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  set(codegen_examples_ ${ARG_EXAMPLES})
  set(backend ${ARG_BACKEND})

  foreach(example ${codegen_examples_})
      cuda_add_executable(${example}_benchmarks_${backend}_cuda "${cwd}/CodeGen/${example}_benchmark.cu" "${cwd}/CodeGen/TestMain.cpp" "${cwd}/CodeGen/Options.cpp")

      target_include_directories(${example}_benchmarks_${backend}_cuda PUBLIC ${include_dirs})
      if(${backend} STREQUAL cuda)
        add_dependencies(${example}_benchmarks_${backend}_cuda gtclang ${example}_c++-naive_codegen ${example}_${backend}_codegen)
      else()
        add_dependencies(${example}_benchmarks_${backend}_cuda gtclang ${example}_c++-naive_codegen ${example}_${backend}_codegen)
      endif()
      if(${backend} STREQUAL gridtools)
        target_compile_definitions(${example}_benchmarks_${backend}_cuda PRIVATE -DBACKEND_CUDA)
      endif()

      target_link_libraries(${example}_benchmarks_${backend}_cuda gtest ${CUDA_CUDART_LIBRARY})
      target_compile_definitions(${example}_benchmarks_${backend}_cuda PUBLIC GRIDTOOLS_CLANG_CUDA ${GTCLANG_DEFINITIONS})
      target_compile_definitions(${example}_benchmarks_${backend}_cuda PRIVATE -DOPTBACKEND=${backend})

      add_custom_target(
        dummy_${example}_${backend}
        DEPENDS ${cwd}/CodeGen/generated/${example}_c++-naive.cpp
      )

      if(CTEST_CUDA_SUBMIT) 
        string(REPLACE " " ";" GTCLANG_SLURM_RESOURCES_LIST ${GTCLANG_SLURM_RESOURCES})
        set(CTEST_SUBMIT_COMMAND "${SLURM_SRUN_COMMAND}" "-n" "${GTCLANG_SLURM_N_TASKS}" ${GTCLANG_SLURM_RESOURCES_LIST} "-p" "${GTCLANG_SLURM_PARTITION}")
      endif(CTEST_CUDA_SUBMIT)

      add_test(NAME CTest-${example}_benchmarks_${backend}_cuda COMMAND ${CTEST_SUBMIT_COMMAND} $<TARGET_FILE:${example}_benchmarks_${backend}_cuda> 12 12 10 --gtest_output=xml:${example}_cuda_unittest.xml)
  endforeach()
endfunction(cuda_compile_generated_examples)


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
  ${cwd}/IIRSerializer
)

list(APPEND compiler_flags -std=c++11 -O2)
list(APPEND library_flags ${Boost_LIBRARIES})

list(APPEND gridtools_flags ${compiler_flags} ${include_flags} ${library_flags})

## The cuda backend is not supported by the python scripts for no-codegen
if(NOT GTCLANG_BUILD_CUDA_EXAMPLES)
  gtclang_add_integrationtest(NAME no-codegen
                            GTCLANG ${GTCLANG_EXECUTABLE}
                            DIRECTORIES ${directories}
                            GRIDTOOLS_FLAGS ${gridtools_flags}
  )
endif(NOT GTCLANG_BUILD_CUDA_EXAMPLES)

# Add a dummy target with all the source files so that they are included in a qt creator project
yoda_add_custom_dummy_target(NAME GTClangQtCreatorParseProjectIntegrationTest
                             DIRECTORIES ${directories})
# Add a dummy target with all the source files so that they are included in a qt creator project
yoda_add_custom_dummy_target(NAME GTClangQtCreatorParseProjectCodeGenTest
                         DIRECTORIES ${cwd}/CodeGen)

# Codegen tests require GridTools
if(GTCLANG_HAS_GRIDTOOLS)
  set(codegen_examples copy_stencil lap conditional_stencil coriolis_stencil hori_diff_stencil_01 hori_diff_stencil_02
      hori_diff_type2_stencil hd_smagorinsky intervals_stencil globals_stencil stencil_functions
      nested_stencil_functions stencil_desc_ast boundary_condition boundary_condition_2 tridiagonal_solve
      kparallel_solver asymmetric)
  # specific examples for cuda backend
  set(cuda_codegen_examples intervals01 intervals02 intervals03 local_kcache kcache_fill kcache_fill_kparallel
      kcache_fill_backward kcache_flush kcache_epflush)

  set(codegen_example_benchmarks)

  ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition_2" "-max-fields=2")
  ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition_2" "-fsplit-stencils")
  
  ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition" "-max-fields=2")
  ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition" "-fsplit-stencils")

  ADD_COMPILER_FLAG_TO_EXAMPLE("intervals02" "-disable-kcaches")
  ADD_COMPILER_FLAG_TO_EXAMPLE("intervals03" "-disable-kcaches")

  ADD_COMPILER_FLAG_TO_EXAMPLE("lap" "-fpass-tmp-to-function")

  set(include_dirs)
  list(APPEND include_dirs "${CMAKE_SOURCE_DIR}/src")
  list(APPEND include_dirs "${GRIDTOOLS_INCLUDE_DIRS}")
  list(APPEND include_dirs "${CMAKE_SOURCE_DIR}")
  list(APPEND include_dirs "${Boost_INCLUDE_DIR}")
  if(GTCLANG_HAS_GRIDTOOLS)
    if(GTCLANG_BUILD_GT_CPU_EXAMPLES OR GTCLANG_BUILD_GT_GPU_EXAMPLES)
      code_generate_examples(EXAMPLES ${codegen_examples} ${cuda_codegen_examples} BACKEND gt)
    endif()
    if(GTCLANG_BUILD_GT_CPU_EXAMPLES)
      compile_generated_examples(EXAMPLES ${codegen_examples} DUMMY ${cuda_codegen_examples} BACKEND gt)
    endif()
    if(GTCLANG_BUILD_GT_GPU_EXAMPLES)
      cuda_compile_generated_examples(EXAMPLES ${codegen_examples} BACKEND gt)
    endif()
    if(GTCLANG_BUILD_CUDA_EXAMPLES)
      code_generate_examples(EXAMPLES ${codegen_examples} ${cuda_codegen_examples} BACKEND cuda)
      cuda_compile_generated_examples(EXAMPLES ${codegen_examples} ${cuda_codegen_examples} BACKEND cuda)
    endif(GTCLANG_BUILD_CUDA_EXAMPLES)
  endif()
endif()
