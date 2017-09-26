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

#.rst:
# dawn_get_compiler_info
# ----------------------
#
# Get the identification string of the compiler.
#
# .. code-block:: cmake
#
#   dawn_get_compiler()
#
# The functions defines the following variable:
#
# ``DAWN_COMPILER_STRING``
#   String of the currently used compiler.
# 
# and conditionally one of the following:
#
# ``DAWN_COMPILER_MSVC``
#   Set to 1 if the compiler in use is MSVC
# ``DAWN_COMPILER_GNU``
#   Set to 1 if the compiler in use is GNU
# ``DAWN_COMPILER_CLANG``
#   Set to 1 if the compiler in use is Clang
#
macro(dawn_get_compiler_info)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    set(DAWN_COMPILER_MSVC 1 CACHE INTERNAL "Compiler is MSVC" FORCE)
    set(DAWN_COMPILER_STRING "Visual Studio (${CMAKE_CXX_COMPILER_VERSION})" 
        CACHE INTERNAL "Compiler-id string" FORCE)
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(DAWN_COMPILER_GNU 1 CACHE INTERNAL "Compiler is GNU gcc" FORCE)
    set(DAWN_COMPILER_STRING "GNU gcc (${CMAKE_CXX_COMPILER_VERSION})" 
        CACHE INTERNAL "Compiler-id string" FORCE)
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(DAWN_COMPILER_CLANG 1 CACHE INTERNAL "Compiler is LLVM Clang" FORCE)
    set(DAWN_COMPILER_STRING "LLVM Clang (${CMAKE_CXX_COMPILER_VERSION})" 
        CACHE INTERNAL "Compiler-id string" FORCE)
  endif()
endmacro()
