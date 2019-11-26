include(CMakeParseArguments)

function(generate_target)
  set(options)
  set(oneValueArgs TEST BACKEND)
  set(multiValueArgs FLAGS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  set(backend ${ARG_BACKEND})
  set(test ${ARG_TEST})

  # Add json input files if they exist
  set(config_str)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${test}.json)
    set(config_str --config=${CMAKE_CURRENT_SOURCE_DIR}/${test}.json)
  endif()

  if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    list(APPEND config_str -fpass-verbose)
  endif()

  list(APPEND config_str ${ARG_FLAGS})

  # Add make target
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/)
  set(generated_file ${CMAKE_CURRENT_BINARY_DIR}/generated/${test}_${backend}.cpp)
  set(source_file ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cpp)
  add_custom_command(OUTPUT ${generated_file}
    COMMAND $<TARGET_FILE:gtclang> -backend=${backend} ${config_str} -o ${generated_file} ${source_file}
    DEPENDS gtclang ${source_file}
  )

  add_custom_target(CodeGen_${test}_${backend}_codegen DEPENDS ${generated_file})
endfunction()

function(compile_target)
  set(options)
  set(oneValueArgs TEST BACKEND)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(backend ${ARG_BACKEND})
  set(test ${ARG_TEST})

  set(executable ${test}_${backend}_test)
  set(sources ${test}_benchmark.cpp TestMain.cpp Options.cpp)

  add_executable(${executable} ${sources})
  add_dependencies(${executable}
    CodeGen_${test}_${backend}_codegen CodeGen_${test}_c++-naive_codegen)
  target_include_directories(${executable} PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR}) #TODO do we need the binary dir?

  if(${backend} STREQUAL gt)
    target_compile_definitions(${executable} PRIVATE -DBACKEND_MC -DGT_ENABLE_METERS)
  endif()

  target_compile_definitions(${executable} PRIVATE -DOPTBACKEND=${backend})
  target_compile_features(${executable} PRIVATE cxx_std_11)
  target_link_libraries(${executable} GTClang GTClangUnittest GridTools::gridtools gtest)

  add_test(NAME ${executable}
    COMMAND $<TARGET_FILE:${executable}> 12 12 10 --gtest_output=xml:${executable}_unittest.xml
  )
endfunction()

function(compile_target_cuda)
#       if(CTEST_CUDA_SUBMIT)
#         string(REPLACE " " ";" GTCLANG_SLURM_RESOURCES_LIST ${GTCLANG_SLURM_RESOURCES})
#         set(CTEST_SUBMIT_COMMAND "${SLURM_SRUN_COMMAND}" "-n" "${GTCLANG_SLURM_N_TASKS}" ${GTCLANG_SLURM_RESOURCES_LIST} "-p" "${GTCLANG_SLURM_PARTITION}")
#       endif(CTEST_CUDA_SUBMIT)

#       add_test(NAME CTest-${example}_benchmarks_${backend}_cuda COMMAND ${CTEST_SUBMIT_COMMAND} $<TARGET_FILE:${example}_benchmarks_${backend}_cuda> 12 12 10 --gtest_output=xml:${example}_cuda_unittest.xml)

  set(options)
  set(oneValueArgs TEST BACKEND)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(backend ${ARG_BACKEND})
  set(test ${ARG_TEST})

  set(executable ${test}_${backend}_cuda_test)
  set(sources ${test}_benchmark.cu TestMain.cpp Options.cpp)

  add_executable(${executable} ${sources})
  add_dependencies(${executable}
    CodeGen_${test}_${backend}_codegen CodeGen_${test}_c++-naive_codegen)

  target_compile_options(${executable} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_ARCH}>)
  if(${backend} STREQUAL gt)
    target_compile_definitions(${executable} PRIVATE -DBACKEND_CUDA)
    target_compile_definitions(${executable} PRIVATE -DGT_ENABLE_METERS)
  endif()
  if(${backend} STREQUAL cuda)
    target_compile_definitions(${executable} PRIVATE -DGRIDTOOLS_CLANG_CUDA)
  endif()

  target_include_directories(${executable} PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

  target_compile_definitions(${executable} PRIVATE -DOPTBACKEND=${backend})
  target_compile_features(${executable} PRIVATE cxx_std_11)
  target_link_libraries(${executable} GridTools::gridtools gtest) # TODO GT only if GT backend

  target_include_directories(${executable}
    PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  )

#TODO
#       if(CTEST_CUDA_SUBMIT)
#         string(REPLACE " " ";" GTCLANG_SLURM_RESOURCES_LIST ${GTCLANG_SLURM_RESOURCES})
#         set(CTEST_SUBMIT_COMMAND "${SLURM_SRUN_COMMAND}" "-n" "${GTCLANG_SLURM_N_TASKS}" ${GTCLANG_SLURM_RESOURCES_LIST} "-p" "${GTCLANG_SLURM_PARTITION}")
#       endif(CTEST_CUDA_SUBMIT)
  add_test(NAME ${executable}
    COMMAND $<TARGET_FILE:${executable}> 12 12 10 --gtest_output=xml:${executable}_unittest.xml
  )
endfunction()

function(add_codegen_test)
  set(options GPU_ONLY)
  set(oneValueArgs TEST)
  set(multiValueArgs FLAGS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  generate_target(TEST ${ARG_TEST} BACKEND c++-naive FLAGS ${ARG_FLAGS})
  if(GTCLANG_BUILD_TESTING_GT_MC OR GTCLANG_BUILD_TESTING_GT_CUDA)
    generate_target(TEST ${ARG_TEST} BACKEND gt FLAGS ${ARG_FLAGS})
  endif()
  if(GTCLANG_BUILD_TESTING_PLAIN_CUDA)
    generate_target(TEST ${ARG_TEST} BACKEND cuda FLAGS ${ARG_FLAGS})
  endif()

  if(NOT ARG_GPU_ONLY AND GTCLANG_BUILD_TESTING_GT_MC)
    compile_target(TEST ${ARG_TEST} BACKEND gt)
  endif()

  # TODO enable GT backend
  # if(GTCLANG_BUILD_TESTING_GT_CUDA)
  #   compile_target_cuda(TEST ${ARG_TEST} BACKEND gt)
  # endif()

  if(GTCLANG_BUILD_TESTING_PLAIN_CUDA)
    compile_target_cuda(TEST ${ARG_TEST} BACKEND cuda)
  endif()
endfunction()

set(CMAKE_CUDA_STANDARD 14)

add_codegen_test(TEST copy_stencil)
add_codegen_test(TEST lap FLAGS -fpass-tmp-to-function)
add_codegen_test(TEST conditional_stencil)
add_codegen_test(TEST coriolis_stencil)
add_codegen_test(TEST hori_diff_stencil_01)
add_codegen_test(TEST hori_diff_stencil_02)
add_codegen_test(TEST hori_diff_type2_stencil)
add_codegen_test(TEST hd_smagorinsky)
add_codegen_test(TEST intervals_stencil)
add_codegen_test(TEST globals_stencil)
add_codegen_test(TEST stencil_functions)
add_codegen_test(TEST nested_stencil_functions)
add_codegen_test(TEST tridiagonal_solve)
add_codegen_test(TEST stencil_desc_ast)
add_codegen_test(TEST kparallel_solver)
add_codegen_test(TEST asymmetric)
# add_codegen_test(TEST boundary_condition FLAGS -max-fields=2 -fsplit-stencils)
# add_codegen_test(TEST boundary_condition_2 FLAGS -max-fields=2 -fsplit-stencils)

# specific examples for cuda backend
add_codegen_test(TEST intervals01 GPU_ONLY)
add_codegen_test(TEST intervals02 FLAGS -disable-kcaches GPU_ONLY)
add_codegen_test(TEST intervals03 FLAGS -disable-kcaches GPU_ONLY)
add_codegen_test(TEST local_kcache GPU_ONLY)
add_codegen_test(TEST kcache_fill GPU_ONLY)
add_codegen_test(TEST kcache_fill_kparallel GPU_ONLY)
add_codegen_test(TEST kcache_fill_backward GPU_ONLY)
add_codegen_test(TEST kcache_flush GPU_ONLY)
add_codegen_test(TEST kcache_epflush GPU_ONLY)
