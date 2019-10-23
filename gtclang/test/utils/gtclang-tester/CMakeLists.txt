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

## gtclang_add_integrationtest
## ---------------------------
##
## Create a target to run integrationtests with the gtclang_tester as well as a bash script.
##
##    NAME:STRING=<>             - Name of the integration test
##    DIRECTORIES:STRING=<>      - List of directories.
##    GTCLANG:PATH=<>            - Path to the GTClang executable
##    GRIDTOOLS_FLAGS:STRING=<>  - List of compile flags needed to compile gridtools C++ code
##
function(gtclang_add_integrationtest)
  # Parse arguments
  set(options)
  set(one_value_args GTCLANG NAME)
  set(multi_value_args DIRECTORIES GRIDTOOLS_FLAGS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(script ${CMAKE_SOURCE_DIR}/test/utils/gtclang-tester/gtclang-tester.py)
  set(command_no_dir ${PYTHON_EXECUTABLE} ${script}
                     --gtclang=${ARG_GTCLANG}
                     --cxx=${CMAKE_CXX_COMPILER})
  set(command ${command_no_dir} ${ARG_DIRECTORIES})

  if(ARG_GRIDTOOLS_FLAGS)
    set(command ${command} --gridtools_flags="${ARG_GRIDTOOLS_FLAGS}")
    set(command_no_dir ${command_no_dir} --gridtools_flags="${ARG_GRIDTOOLS_FLAGS}")
  endif()

  string(REPLACE ";" " " cmd_script "${command}")
  file(WRITE ${CMAKE_BINARY_DIR}/gtclang-tester-${ARG_NAME}.sh "#!/bin/bash\n${cmd_script} $*\n")

  string(REPLACE ";" " " cmd_script_no_dir "${command_no_dir}")
  file(WRITE ${CMAKE_BINARY_DIR}/gtclang-tester.sh "#!/bin/bash\n${cmd_script_no_dir} $*\n")

  # Add make target
  add_custom_target(
    integrationtest-${ARG_NAME}
    COMMAND ${BASH_EXECUTABLE} ${CMAKE_BINARY_DIR}/gtclang-tester-${ARG_NAME}.sh
    DEPENDS gtclang
  )

  # Add CTest target
  add_test(NAME CTest-IntegrationTest-${ARG_NAME}
           COMMAND ${BASH_EXECUTABLE} ${CMAKE_BINARY_DIR}/gtclang-tester-${ARG_NAME}.sh)
endfunction()

