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
  # PassSetBoundaryCondition
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
