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

include(DawnAddUnittest)

# gtclang_add_unittest
# --------------------
#
# Compile the given objects into a runnable unittest executable (.exe) and register it within CTest.
# The executable will be stored in ${CMAKE_BINARY_DIR}/bin/unittest.
#
#    NAME:STRING=<>     - Name of the unittest exectuable as well as the CMake target to build it.
#    SOURCES:STRING=<>   - List of source files making up the exectuable.
#
macro(gtclang_add_unittest)
  cmake_parse_arguments(ARG "" "NAME" "SOURCES" ${ARGN})

  if(NOT("${ARG_UNPARSED_ARGUMENTS}" STREQUAL ""))
    message(FATAL_ERROR "gtclang_add_unittest: invalid argument ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  dawn_add_unittest(
    NAME ${ARG_NAME}
    SOURCES ${ARG_SOURCES}
    DEPENDS GTClangStatic ${DAWN_EXTERNAL_LIBRARIES} gtest
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/unittest
    GTEST_ARGS --gtest_color=yes
  )
endmacro()

set(GTCLANG_UNITTEST_EXTERNAL_LIBRARIES 
  GTClangStatic
  GTClangUnittestStatic
  ${GTCLANG_EXTERNAL_LIBRARIES}
)

add_subdirectory(gtclang)
