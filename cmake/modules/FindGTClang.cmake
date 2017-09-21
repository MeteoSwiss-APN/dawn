##===-----------------------------------------------------------------------------*- Python -*-===##
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

#.rst:
# FindGTClang
# -----------
#
# This script locates the gridtools clang DSL tools. This script makes use of the standard 
# find_package arguments of ``VERSION``, ``REQUIRED`` and ``QUIET``. GTCLANG_FOUND will report if 
# an acceptable version of gtclang was found.
#
# The script will prompt the user to specify GTCLANG_ROOT if the prefix cannot be determined by the 
# location of gtclang in the system path and REQUIRED is specified to find_package().
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# Defines the following variables:
#
#  GTCLANG_FOUND                  - True if gtclang was found
#  GTCLANG_EXECUTABLE             - Path to ``gtclang``
#  GTCLANG_INCLUDE_DIRS           - Include directory for gridtools_clang runtime headers. 
#
# Macros
# ^^^^^^
#
# The script creates the following macros:
#
#   GTCLANG_COMPILE( generated_files file0 file1 ... )
#   -- Returns a list of generated files from the input source files to be used
#      with ``add_executable`` or ``add_library``
#      For example to compile foo and bar:
#
#         gtclang_compile(generated_files foo.cpp bar.cpp)
#         add_executable(foo-bar ${generated_files})
#
#      Set ``GTCLANG_FLAGS`` to pass custom arguments to gtclang. 
#
# Hints
# ^^^^^
#
# Set ``GTCLANG_ROOT`` to a directory that contains a gtclang installation, or directly set 
# ``GTCLANG_EXECUTABLE`` and ``GTCLANG_INCLUDE_DIRS``.
#

include(FindPackageHandleStandardArgs)

#
# Parse GTCLANG_ROOT
#
set(GTCLANG_ROOT_ENV $ENV{GTCLANG_ROOT})
if(GTCLANG_ROOT_ENV)
  set(GTCLANG_ROOT ${GTCLANG_ROOT_ENV} CACHE PATH "gtclang install path.")
endif()

if(NOT(DEFINED GTCLANG_ROOT))
  find_path(GTCLANG_ROOT NAMES include/gridtools/clang_dsl.hpp HINTS ${GTCLANG_INCLUDE_DIRS})
else()
  get_filename_component(_GTCLANG_ROOT_ABSOLUTE ${GTCLANG_ROOT} ABSOLUTE)
  set(GTCLANG_ROOT ${_GTCLANG_ROOT_ABSOLUTE} CACHE PATH "gtclang install path.")
endif()

#===---------------------------------------------------------------------------------------------===
#   Find gridtools/clang runtime headers
#====--------------------------------------------------------------------------------------------===
if(NOT(DEFINED GTCLANG_INCLUDE_DIRS))
  find_path(GTCLANG_INCLUDE_DIRS NAMES gridtools/clang_dsl.hpp HINTS ${GTCLANG_ROOT}/include)
endif()

#===---------------------------------------------------------------------------------------------===
#   Find gtclang
#====--------------------------------------------------------------------------------------------===
if(NOT(DEFINED GTCLANG_EXECUTABLE))
  find_file(GTCLANG_EXECUTABLE gtclang ${GTCLANG_ROOT}/bin/gtclang)
endif()

macro(gtclang_compile generated_files)
  unset(_generated_files)
  foreach(file ${ARGN})
    get_filename_component(absolute_path ${file} ABSOLUTE)
    file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ${absolute_path})

    get_filename_component(output_dir ${relative_path} DIRECTORY)
    get_filename_component(filename ${file} NAME_WE)
    get_filename_component(ext ${file} EXT)

    set(output_file "${CMAKE_BINARY_DIR}/${output_dir}/${filename}_gen${ext}")
    set(output_file_relative "${output_dir}/${filename}_gen${ext}")

    unset(_cmd)
    list(APPEND _cmd ${GTCLANG_EXECUTABLE} ${absolute_path} -I${GTCLANG_INCLUDE_DIRS} 
                     -o${output_file})
    foreach(flag ${GTCLANG_FLAGS})
        list(APPEND _cmd ${flag})
    endforeach()
    
    add_custom_command(OUTPUT ${output_file}
                       COMMAND ${_cmd}
                       DEPENDS ${file}
                       COMMENT "Building GTClang object ${output_file_relative}")
    list(APPEND _generated_files ${output_file})
    unset(_cmd)
  endforeach()
  set(${generated_files} ${_generated_files})
endmacro()

#===---------------------------------------------------------------------------------------------===
#   Report result 
#====--------------------------------------------------------------------------------------------===
find_package_handle_standard_args(
  GTClang
  FAIL_MESSAGE "Could NOT find gtclang. (Try setting GTCLANG_ROOT in the env)"
  REQUIRED_VARS  GTCLANG_INCLUDE_DIRS
                 GTCLANG_EXECUTABLE
)

mark_as_advanced(GTCLANG_EXECUTABLE
                 GTCLANG_INCLUDE_DIRS)

if(GTCLANG_FOUND)
  message(STATUS "GTClang found: ${GTCLANG_EXECUTABLE}")
else()
  if(${GTClang_FIND_REQUIRED})
     message(FATAL_ERROR "Could NOT find gtclang. (Try setting GTCLANG_ROOT in the env)")
  endif()
endif()
