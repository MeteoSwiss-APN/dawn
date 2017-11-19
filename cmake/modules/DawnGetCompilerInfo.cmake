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

include(msbuildIncludeGuard)
msbuild_include_guard()

#.rst:
# msbuid_get_compiler_info
# ----------------------
#
# Get the identification string of the compiler.
#
# .. code-block:: cmake
#
#   msbuild_get_compiler()
#
# The functions defines the following variable:
#
# ``MSBUILD_COMPILER_STRING``
#   String of the currently used compiler.
# 
# and conditionally one of the following:
#
# ``MSBUILD_COMPILER_MSVC``
#   Set to 1 if the compiler in use is MSVC
# ``MSBUILD_COMPILER_GNU``
#   Set to 1 if the compiler in use is GNU
# ``MSBUILD_COMPILER_CLANG``
#   Set to 1 if the compiler in use is Clang
#
macro(msbuild_get_compiler_info)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    set(MSBUILD_COMPILER_MSVC 1 CACHE INTERNAL "Compiler is MSVC" FORCE)
    set(MSBUILD_COMPILER_STRING "Visual Studio (${CMAKE_CXX_COMPILER_VERSION})" 
        CACHE INTERNAL "Compiler-id string" FORCE)
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(MSBUILD_COMPILER_GNU 1 CACHE INTERNAL "Compiler is GNU gcc" FORCE)
    set(MSBUILD_COMPILER_STRING "GNU gcc (${CMAKE_CXX_COMPILER_VERSION})" 
        CACHE INTERNAL "Compiler-id string" FORCE)
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(MSBUILD_COMPILER_CLANG 1 CACHE INTERNAL "Compiler is LLVM Clang" FORCE)
    set(MSBUILD_COMPILER_STRING "LLVM Clang (${CMAKE_CXX_COMPILER_VERSION})" 
        CACHE INTERNAL "Compiler-id string" FORCE)
  endif()
endmacro()
