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

gtclang_include_guard()

include(CMakeParseArguments)

## gtclang_report_result
## ---------------------
##
## Report a list of strings as STATUS.
##
##    FILE:STRING=<>             - Header of the current report.
##    ARGN:STRING=<>             - List of strings to report.
##
function(gtclang_report_result HEADER)
  string(LENGTH ${HEADER} header_length)
  set(full_header "----------------------------------------------------------------")
  math(EXPR right_header_length "43 - ${header_length}")
  string(SUBSTRING ${full_header} "0" "${right_header_length}" right_header)
  
  message(STATUS "------------------- ${HEADER} ${right_header}")
  foreach(arg ${ARGN})
    message(STATUS "${arg}")
  endforeach()
  message(STATUS "${full_header}")
endfunction()

## gtclang_add_library
## -------------------
##
## Create a CMake-Object library ("<NAME>Objects") aswell as a static library ("<NAME>Static>").
## All header files (".h") in the directory of the source files are treated as dependencies of the
## library.
##
## Note that this function does provide an install traget, invoke gtclang_combine_libraries on the 
## object libraries.
##    
##    NAME:STRING=<>             - Name of the library.
##    SOURCES:STRING=<>          - List of source files.
##    DEPENDS:STRING=<>          - List of external libraries and/or CMake targets treated as
##                                 dependencies of the library.
##
function(gtclang_add_library)
  #
  # Parse arguments
  #
  set(options)
  set(one_value_args NAME)
  set(multi_value_args SOURCES DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  
  if(NOT("${ARG_UNPARSED_ARGUMENTS}" STREQUAL ""))
    message(FATAL_ERROR "invalid argument ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  #
  # Add static library
  #
  add_library(${ARG_NAME} STATIC ${ARG_SOURCES})
  target_link_libraries(${ARG_NAME} ${ARG_DEPENDS})
endfunction()

## gtclang_add_executable
## ----------------------
##
## Create an executable.
## All header files (".h") in the directory of the source files are treated as dependencies of the
## library.
##    
##    NAME:STRING=<>             - Name of the test.
##    SOURCES:STRING=<>          - List of source files.
##    DEPENDS:STRING=<>          - List of external libraries and/or CMake targets to link against.
##
function(gtclang_add_executable)
  #
  # Parse arguments
  #
  set(one_value_args NAME)
  set(multi_value_args SOURCES DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  
  if(NOT("${ARG_UNPARSED_ARGUMENTS}" STREQUAL ""))
    message(FATAL_ERROR "invalid argument ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  
  #
  # Add executable
  #
  add_executable(${ARG_NAME} ${ARG_SOURCES})
  target_link_libraries(${ARG_NAME} ${ARG_DEPENDS}) 
  set_target_properties(${ARG_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

## gtclang_add_unittest
## --------------------
##
## Create a unittest executable and register it within CTest.
## All header files (".h") in the directory of the source files are treated as dependencies of the
## library.
##    
##    NAME:STRING=<>             - Name of the unittest.
##    SOURCES:STRING=<>          - List of source files.
##    DEPENDS:STRING=<>          - List of external libraries and/or CMake targets to link against.
##
function(gtclang_add_unittest)
  #
  # Parse arguments
  #
  set(options)
  set(one_value_args NAME)
  set(multi_value_args SOURCES DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  
  #
  # Add executable
  #
  gtclang_add_executable(${ARGN})
  
  #
  # Add test
  #
  add_test(NAME CTest-${ARG_NAME} COMMAND $<TARGET_FILE:${ARG_NAME}> --gtest_color=yes)
endfunction()

## gtclang_configure_file
## ----------------------
##
## Configure a file and store the output in the same directory hirarchy as the input file while 
## substituting ${CMAKE_SOURCE_DIR} for ${CMAKE_BINARY_DIR}.
##
## If the input file has a ".cmake" extension, it will be removed in the output.
##
## For Example: gtclang_configure_file(${CMAKE_SOURCE_DIR}/src/gtclang/Config.h.cmake)
## will store the configured file in "${CMAKE_BINARY_DIR}/src/gtclang/Config.h".
##
##    FILE:STRING=<>             - File to configure.
##
function(gtclang_configure_file FILE)
  get_filename_component(absolute_path ${FILE} ABSOLUTE)
  file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ${absolute_path})

  get_filename_component(output_dir ${relative_path} DIRECTORY)
  get_filename_component(output_name_cmake ${relative_path} NAME)
  string(REGEX REPLACE ".cmake" "" output_file_name ${output_name_cmake})
  
  set(input_file ${FILE})
  set(output_file "${CMAKE_BINARY_DIR}/${output_dir}/${output_file_name}")
  
  configure_file(${input_file} ${output_file})
endfunction(gtclang_configure_file)

