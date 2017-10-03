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

include(DawnGetScriptDir)

#.rst:
# dawn_get_git_head_revision
# --------------------------
#
# These functions force a re-configure on each git commit so that you can trust the values of the 
# variables in your build system. It returns the refspec and sha1 hash of the current head revision.
#
# .. code-block:: cmake
#
#   dawn_get_git_head_revision(HEAD_REF HEAD_HASH)
# 
# ``HEAD_REF``
#   The `Refspec <https://git-scm.com/book/en/v2/Git-Internals-The-Refspec>`_.
# ``HEAD_HASH``
#   sha1 hash of the current head revision.
#
function(dawn_get_git_head_revision HEAD_REF HEAD_HASH)
  set(git_parent_dir "${CMAKE_CURRENT_SOURCE_DIR}")
  set(git_dir "${git_parent_dir}/.git")

  # Search for .git directory
  while(NOT EXISTS "${git_dir}")
    set(git_prev_parent_dir "${git_parent_dir}")
    get_filename_component(git_parent_dir ${git_parent_dir} PATH)
    if(git_parent_dir STREQUAL git_prev_parent_dir)
      # We have reached the root directory, we are not in git
      set(${HEAD_REF} "GITDIR-NOTFOUND" PARENT_SCOPE)
      set(${HEAD_HASH} "GITDIR-NOTFOUND" PARENT_SCOPE)
      return()
    endif()
    set(git_dir "${git_parent_dir}/.git")
  endwhile()

  # check if this is a submodule
  if(NOT IS_DIRECTORY ${git_dir})
    file(READ ${git_dir} submodule)
    string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" git_dir_RELATIVE ${submodule})
    get_filename_component(submodule_dir ${git_dir} PATH)
    get_filename_component(git_dir ${submodule_dir}/${git_dir_RELATIVE} ABSOLUTE)
  endif()

  # Create directory git-data in the binary dir
  set(git_data_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/git-data")
  if(NOT EXISTS "${git_data_dir}")
    file(MAKE_DIRECTORY "${git_data_dir}")
  endif()

  if(NOT EXISTS "${git_dir}/HEAD")
    return()
  endif()

  # Copy .git/HEAD into CMakeFiles/git-data
  set(head_file ${git_data_dir}/HEAD)
  configure_file("${git_dir}/HEAD" "${head_file}" COPYONLY)

  dawn_get_script_dir(script_dir)
  set(input_script ${script_dir}/DawnGetGitHeadRevision-Script.cmake.in)
  set(output_script ${CMAKE_BINARY_DIR}/CMakeFiles/git-data/cmake/DawnGetGitHeadRevision-Script.cmake)

  # Configure the script
  set(in_head_file ${head_file})
  set(in_git_dir ${git_dir})
  set(in_git_data_dir ${git_data_dir})
  configure_file(${input_script} ${output_script} @ONLY)
  include(${output_script})

  set(${HEAD_REF} "${GIT_HEAD_REF}" PARENT_SCOPE)
  set(${HEAD_HASH} "${GIT_HEAD_HASH}" PARENT_SCOPE)
endfunction()
