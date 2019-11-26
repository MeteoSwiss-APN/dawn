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

add_subdirectory(CodeGen)

set(dummy_source ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dummy.cpp)
file(WRITE ${dummy_source} "int main() { return 0; }")

set(directories
  Accesses
  Diagnostics
  PassStageSplitter
  PassMultiStageSplitter
  PassSetCaches
  PassStageReordering
  PassStageMerger
  PassFieldVersioning
  PassTemporaryMerger
  PassTemporaryType
  PassPreprocessor
  PassSetNonTempCaches
  PassSetBoundaryCondition
  SIR
  IIRSerializer
  Regression
)

# The cuda backend is not supported by the python scripts for no-codegen
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(command ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/utils/gtclang-tester/gtclang-tester.py
  --gtclang=$<TARGET_FILE:gtclang>
  --cxx=${CMAKE_CXX_COMPILER})

foreach(dir ${directories})
  set(command ${command} ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
endforeach()

list(APPEND gridtools_flags ${compiler_flags} ${include_flags} ${library_flags})

set(command ${command} --gridtools_flags="${gridtools_flags}")

string(REPLACE ";" " " cmd_script "${command}")
file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/gtclang-tester-no-codegen.sh CONTENT "#!/bin/bash\n${cmd_script} $*\n")

# Add make target
add_custom_target(
  integrationtest-no-codegen
  COMMAND ${BASH_EXECUTABLE} ${PROJECT_BINARY_DIR}/gtclang-tester-no-codegen.sh
  DEPENDS gtclang
  )

# Add CTest target
find_program (BASH_EXECUTABLE bash)
add_test(NAME CTest-IntegrationTest-no-codegen
  COMMAND ${BASH_EXECUTABLE} ${PROJECT_BINARY_DIR}/gtclang-tester-no-codegen.sh)

# list(APPEND compiler_flags -std=c++11 -O2)
# list(APPEND library_flags ${Boost_LIBRARIES})

# # examples from FV3
# set(fv3_tests p_grad_c)

# function(compile_generated_tests)
#   set(oneValueArgs BACKEND DIRECTORY)
#   set(multiValueArgs TESTS DUMMY)
#   cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

#   set(backend ${ARG_BACKEND})

#   if(NOT ${backend} STREQUAL "gt")
#     message(FATAL "backend ${backend} not supported")
#   endif()

#   foreach(example ${ARG_DUMMY})
#     add_executable(${example}_benchmarks_${backend} ${cwd}/${ARG_DIRECTORY}/dummy.cpp)
#     set_property(SOURCE ${cwd}/${ARG_DIRECTORY}/dummy.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/${ARG_DIRECTORY}/generated/${example}_c++-naive.cpp)
#     if(GTCLANG_BUILD_CUDA_TESTS)
#       set_property(SOURCE ${cwd}/${ARG_DIRECTORY}/dummy.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/${ARG_DIRECTORY}/generated/${example}_cuda.cpp)
#     endif(GTCLANG_BUILD_CUDA_TESTS)
#   endforeach()
#   foreach(example ${ARG_TESTS})
#     add_executable(${example}_benchmarks_${backend} "${cwd}/${ARG_DIRECTORY}/${example}_benchmark.cpp" "${cwd}/${ARG_DIRECTORY}/TestMain.cpp" "${cwd}/${ARG_DIRECTORY}/Options.cpp")
#     target_include_directories(${example}_benchmarks_${backend} PUBLIC ${include_dirs})
#     target_compile_definitions(${example}_benchmarks_${backend} PRIVATE -DOPTBACKEND=${backend})
#     target_compile_features(${example}_benchmarks_${backend} PRIVATE cxx_std_11)
#     if(${backend} STREQUAL gt)
#       target_compile_definitions(${example}_benchmarks_${backend} PRIVATE -DBACKEND_MC)
#       target_compile_definitions(${example}_benchmarks_${backend} PRIVATE -DGT_ENABLE_METERS)
#     endif()
#     add_dependencies(${example}_benchmarks_${backend} gtclang ${example}_c++-naive_codegen ${example}_${backend}_codegen)
#     target_link_libraries(${example}_benchmarks_${backend} ${GTCLANG_UNITTEST_EXTERNAL_LIBRARIES} gtest)

#     #      set_property(SOURCE ${cwd}/${ARG_DIRECTORY}/${example}_benchmark.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/${ARG_DIRECTORY}/generated/${example}_${backend}.cpp)
#     #      set_property(SOURCE ${cwd}/${ARG_DIRECTORY}/${example}_benchmark.cpp APPEND PROPERTY OBJECT_DEPENDS ${cwd}/${ARG_DIRECTORY}/generated/${example}_c++-naive.cpp)

#     add_test(NAME CTest-${example}_benchmarks_${backend} COMMAND $<TARGET_FILE:${example}_benchmarks_${backend}> 12 12 10 --gtest_output=xml:${example}_unittest.xml)

#   endforeach()
# endfunction(compile_generated_tests)

# function(cuda_compile_generated_tests)
#   set(oneValueArgs BACKEND DIRECTORY)
#   set(multiValueArgs TESTS)
#   cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

#   foreach(example ${ARG_TESTS})
#     set(test_name ${example}_benchmarks_${ARG_BACKEND}_cuda)
#     add_executable(${test_name}
#       ${cwd}/${ARG_DIRECTORY}/${example}_benchmark.cu
#       ${cwd}/${ARG_DIRECTORY}/TestMain.cpp
#       ${cwd}/${ARG_DIRECTORY}/Options.cpp
#     )

#     target_compile_features(${test_name} PRIVATE cxx_std_11)
#     target_link_libraries(${test_name} gtest)
#     add_dependencies(${test_name} gtclang ${example}_c++-naive_codegen ${example}_${ARG_BACKEND}_codegen)
#     if(${ARG_BACKEND} STREQUAL gt)
#       target_compile_definitions(${example}_benchmarks_${ARG_BACKEND}_cuda PRIVATE -DBACKEND_CUDA -DGT_ENABLE_METERS)
#     endif()

#     # target_link_libraries(${example}_benchmarks_${backend}_cuda gtest ${CUDA_CUDART_LIBRARY})
#     target_compile_definitions(${test_name} PUBLIC GRIDTOOLS_CLANG_CUDA ${GTCLANG_DEFINITIONS})
#     target_compile_definitions(${test_name} PRIVATE -DOPTBACKEND=${ARG_BACKEND})

#     if(CTEST_CUDA_SUBMIT)
#       string(REPLACE " " ";" GTCLANG_SLURM_RESOURCES_LIST ${GTCLANG_SLURM_RESOURCES})
#       set(CTEST_SUBMIT_COMMAND "${SLURM_SRUN_COMMAND}" "-n" "${GTCLANG_SLURM_N_TASKS}" ${GTCLANG_SLURM_RESOURCES_LIST} "-p" "${GTCLANG_SLURM_PARTITION}")
#     endif(CTEST_CUDA_SUBMIT)

#     add_test(NAME CTest-${test_name}
#       COMMAND ${CTEST_SUBMIT_COMMAND}
#         $<TARGET_FILE:${test_name}> 12 12 10 --gtest_output=xml:${example}_cuda_unittest.xml)
#   endforeach()
# endfunction(cuda_compile_generated_tests)


# set(cwd ${CMAKE_CURRENT_LIST_DIR})

# set(codegen_example_benchmarks)

# ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition_2" "-max-fields=2")
# ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition_2" "-fsplit-stencils")

# ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition" "-max-fields=2")
# ADD_COMPILER_FLAG_TO_EXAMPLE("boundary_condition" "-fsplit-stencils")

# ADD_COMPILER_FLAG_TO_EXAMPLE("intervals02" "-disable-kcaches")
# ADD_COMPILER_FLAG_TO_EXAMPLE("intervals03" "-disable-kcaches")

# ADD_COMPILER_FLAG_TO_EXAMPLE("lap" "-fpass-tmp-to-function")

# # set(include_dirs)
# # list(APPEND include_dirs "${CMAKE_SOURCE_DIR}/src")
# # list(APPEND include_dirs "${CMAKE_SOURCE_DIR}")
# # list(APPEND include_dirs "${Boost_INCLUDE_DIR}")

# # list(APPEND include_dirs "${GridTools_INCLUDE_PATH}")
# if(BUILD_CPU_TESTS OR BUILD_GPU_TESTS)
#   code_generate_tests(TESTS ${codegen_tests} ${cuda_codegen_tests}
#     BACKEND gt
#     DIRECTORY CodeGen)
#   code_generate_tests(TESTS ${fv3_tests}
#     BACKEND gt
#     DIRECTORY FV3)
# endif()
# # if(BUILD_CPU_TESTS)
# #   compile_generated_tests(TESTS ${codegen_tests}
# #     DUMMY ${cuda_codegen_tests}
# #     BACKEND gt
# #     DIRECTORY CodeGen)
# #   compile_generated_tests(TESTS ${fv3_tests}
# #     BACKEND gt
# #     DIRECTORY FV3)
# # endif()
# # if(BUILD_CPU_TESTS)
# #   cuda_compile_generated_tests(TESTS ${codegen_tests}
# #     BACKEND gt
# #     DIRECTORY CodeGen)
# #   cuda_compile_generated_tests(TESTS ${fv3_tests}
# #     BACKEND gt
# #     DIRECTORY FV3)
# # endif()
# # if(BUILD_CUDA_TESTS)
# #   code_generate_tests(TESTS ${codegen_tests} ${cuda_codegen_tests}
# #     BACKEND cuda
# #     DIRECTORY CodeGen)
# #   cuda_compile_generated_tests(TESTS ${codegen_tests} ${cuda_codegen_tests}
# #     BACKEND cuda
# #     DIRECTORY CodeGen)
# # endif()
