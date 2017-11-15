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

# dawn_clone_repository
# ----------------------------
#
# This will make sure the repository NAME exists and, if not, will clone the branch BRANCH 
# from the git repository given by URL.
#
# This will define the variable DAWN_<NAME>_SOURCE_DIR (where <NAME> is the passed NAME 
# in all uppercase) which contains the path to the source of the repository NAME.
#
#    NAME:STRING=<>       - Name of the repository
#    URL:STRING=<>        - Version of the package
#    BRANCH:STRING=<>     - Do we use the system version of the package?
#
function(dawn_clone_repository)
  set(options)
  set(one_value_args NAME URL BRANCH)
  set(multi_value_args)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  string(TOUPPER ${ARG_NAME} upper_name)
  set(source_dir "${CMAKE_SOURCE_DIR}/${ARG_NAME}")

  # Check if repository exists
  if(EXISTS "${source_dir}")
    set("DAWN_${upper_name}_SOURCE_DIR" "${source_dir}" 
         CACHE INTERNAL "source directory of ${ARG_NAME}" FORCE)
    message(STATUS "Using ${ARG_NAME}: ${DAWN_${upper_name}_SOURCE_DIR}")
  else()
    find_program(git_executable "git")
    if(NOT(git_executable))
      message(FATAL_ERROR "ERROR: git not FOUND!")
    endif()
    mark_as_advanced(git_executable)

    macro(run_git)
      unset(result)
      unset(out_var)
      unset(error_var)
      set(cmd "${git_executable}")
      foreach(arg ${ARGN})
        set(cmd ${cmd} "${arg}")
      endforeach()

      execute_process(
        COMMAND ${cmd}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE out_var
        ERROR_VARIABLE error_var
      )

      if(NOT("${result}" STREQUAL "0"))
        string(REPLACE ";" " " cmd_str "${cmd}")
        message(FATAL_ERROR "${error_var}\n\nERROR: failed to run\n\"${cmd_str}\"\n\n${error_var}")
      endif()
    endmacro()
    
    # Clone repository
    message(STATUS "Cloning ${ARG_NAME} from git: ${ARG_URL} ...")
    run_git("clone" "${ARG_URL}" "--branch=${ARG_BRANCH}" ${source_dir})
    message(STATUS "Successfully cloned ${ARG_NAME}")

    set("DAWN_${upper_name}_SOURCE_DIR" "${source_dir}" 
         CACHE INTERNAL "source directory of ${ARG_NAME}" FORCE)
  endif()
endfunction()
