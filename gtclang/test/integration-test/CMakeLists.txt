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

list(TRANSFORM directories PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# NOTE The cuda backend is not supported by the python scripts for no-codegen
add_test(NAME GTClangIntegrationTestNoCodegen
  COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/test/utils/gtclang-tester/gtclang-tester.py
  --gtclang=$<TARGET_FILE:gtclang>
  --cxx=${CMAKE_CXX_COMPILER}
  --gridtools_flags="${compiler_flags} ${include_flags} ${library_flags}"
  ${directories})

set_property(TEST GTClangIntegrationTestNoCodegen APPEND PROPERTY DEPENDS gtclang)
