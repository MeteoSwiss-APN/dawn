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

include(DawnCheckAndSetCXXFlag)

# gtclang_set_cxx_flags
# ---------------------
#
# Set GTClang specific C++ flags.
#
macro(gtclang_set_cxx_flags)
  # Remove -DNDEBUG flag if ASSERTS are ON
  if(GTCLANG_ASSERTS)
    set(ndebug_flag "-DNDEBUG")
    string(REPLACE "${ndebug_flag}" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "${ndebug_flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  endif()
  
  dawn_check_and_set_cxx_flag("-march=native" HAVE_GCC_MARCH_NATIVE)
  dawn_check_and_set_cxx_flag("-Wall" HAVE_GCC_WALL)
  dawn_check_and_set_cxx_flag("-Wno-sign-compare" HAVE_GCC_WNO_SIGN_COMPARE)
  dawn_check_and_set_cxx_flag("-Werror=return-type" HAVE_GCC_ERROR_RETURN_TYPE)
endmacro()