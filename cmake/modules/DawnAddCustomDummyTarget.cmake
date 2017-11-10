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

include(DawnIncludeGuard)
dawn_include_guard()

include(CMakeParseArguments)

## gtclang_add_integrationtest
## ---------------------------
##
## Creates a dummy target with list of C++ files contained in the list of provided directories, with 
##   the purpose that qt creator can load those files into the project, which does not happen for those
##   files that are not included in at least one target.
##    
##    NAME:STRING=<>             - Name of the dummy custom target
##    DIRECTORIES:STRING=<>      - List of directories that contain the source files
##
function(dawn_add_custom_dummy_target)
  # Parse arguments
  set(options)
  set(one_value_args NAME)
  set(multi_value_args DIRECTORIES)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(all_files "")
  foreach(dir ${ARG_DIRECTORIES})

    file(GLOB test_sources "${dir}/*.cpp" "${dir}/*.hpp" "${dir}/*.h")

    list(APPEND all_files ${test_sources} ${test_headers})
  endforeach(dir)

  add_custom_target(${ARG_NAME} ALL SOURCES ${all_files})
endfunction()

