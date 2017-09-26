##===------------------------------------------------------------------------------*- CMake -*-===##
##                          _                      
##                         | |                     
##                       __| | __ ___      ___ __  
##                      / _` |/ _` \ \ /\ / / '_ \ 
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

#.rst:
# dawn_get_git_head_revision
# --------------------------
#
# These functions force a re-configure on each git commit so that you can trust the values of the 
# variables in your build system. It returns the refspec and sha1 hash of the current head revision.
#
# .. code-block:: cmake
#
#   dawn_get_git_head_revision(REF_SPEC HASH_VAR)
# 
# ``REF_SPEC``
#   Variable which will contain the script directory on output.
# ``HASH_VAR``
#   sha1 hash of the current head revision.
#
function(dawn_get_git_head_revision)

endfunction()
