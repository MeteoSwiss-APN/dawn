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

include(CMakeParseArguments)

# gtclang_clone_repository
# ----------------------------
#
# This will make sure the repository NAME exists and, if not, will clone the branch BRANCH 
# from the git repository given by URL.
#
# This will define the variable MCHBUILD_<NAME>_SOURCE_DIR (where <NAME> is the passed NAME 
# in all uppercase) which contains the path to the source of the repository NAME.
#
#    NAME:STRING=<>       - Name of the repository
#    URL:STRING=<>        - Version of the package
#    BRANCH:STRING=<>     - Do we use the system version of the package?
#
function(gtclang_add_dependency)
  set(options)
  set(one_value_args NAME URL BRANCH)
  set(multi_value_args)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  string(TOUPPER ${ARG_NAME} upper_name)
  set(source_dir "${CMAKE_SOURCE_DIR}/${ARG_NAME}")

  include("thirdparty/gtclangAdd${ARG_NAME}")
endfunction()
