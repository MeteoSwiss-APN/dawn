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

include(DawnCheckAndSetCXXFlag)

# dawn_set_cxx_flags
# ------------------
#
# Set project specific C++ flags.
#
macro(dawn_set_cxx_flags)
  # Remove -DNDEBUG flag if ASSERTS are ON
  if(GTCLANG_ASSERTS)
    set(ndebug_flag "-DNDEBUG")
    string(REPLACE "${ndebug_flag}" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "${ndebug_flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  endif()

  # Architecture
  dawn_check_and_set_cxx_flag("-march=native" HAVE_GCC_MARCH_NATIVE)
  
  # Pthread
  dawn_check_and_set_cxx_flag("-pthread" HAVE_GCC_PTHREAD)

  # Warnings
  dawn_check_and_set_cxx_flag("-Wall" HAVE_GCC_WALL)
  dawn_check_and_set_cxx_flag("-W" HAVE_GCC_W)
  dawn_check_and_set_cxx_flag("-Wformat" HAVE_GCC_WFORMAT)
  dawn_check_and_set_cxx_flag("-Wmissing-declarations" HAVE_GCC_WMISSING_DECLARATIONS)
  dawn_check_and_set_cxx_flag("-Wmissing-prototypes" HAVE_GCC_WMISSING_PROTOTYPES)
  dawn_check_and_set_cxx_flag("-Wstrict-prototypes" HAVE_GCC_WSTRICT_PROTOTYPES)
  dawn_check_and_set_cxx_flag("-Wundef" HAVE_GCC_WUNDEF)
  dawn_check_and_set_cxx_flag("-Winit-self" HAVE_GCC_WINI_SELF)
  dawn_check_and_set_cxx_flag("-Wpointer-arith" HAVE_GCC_WPOINTER_ARITH)
  dawn_check_and_set_cxx_flag("-Wshadow" HAVE_GCC_WSHADOW)
  dawn_check_and_set_cxx_flag("-Wsign-promo" HAVE_GCC_WSIGN_PROMO)
  dawn_check_and_set_cxx_flag("-Werror=return-type" HAVE_GCC_WERROR_RETURN_TYPE)
  dawn_check_and_set_cxx_flag("-Werror=non-virtual-dtor" HAVE_GCC_WERROR_NON_VIRTUAL_DTOR)
  dawn_check_and_set_cxx_flag("-Werror=address" HAVE_GCC_WERROR_ADDRESS)
  dawn_check_and_set_cxx_flag("-Werror=sequence-point" HAVE_GCC_WERROR_SEQUENCE_POINT)

  dawn_check_and_set_cxx_flag("-Wno-sign-promo" HAVE_GCC_WNO_SIGN_PROMO)
  dawn_check_and_set_cxx_flag("-Wno-sign-compare" HAVE_GCC_WNO_SIGN_COMPARE)
  dawn_check_and_set_cxx_flag("-Wno-unused-parameter" HAVE_GCC_WNO_UNUSDED_PARAMETER)
  
  if(BUILD_SHARED_LIBS)
    dawn_check_and_set_cxx_flag("-fPIC" HAVE_GCC_PIC)
  endif()

  if(DAWN_USE_CCACHE)
    if(DAWN_COMPILER_CLANG)
      dawn_check_and_set_cxx_flag("-Qunused-arguments" HAVE_CLANG_UNUSED_ARGUMENTS)
      dawn_check_and_set_cxx_flag("-fcolor-diagnostics" HAVE_CLANG_COLOR_DIAGNOSTICS)
      dawn_check_and_set_cxx_flag("-Wno-undefined-var-template" 
                                     HAVE_CLANG_WNO_UNDEFINED_VAR_TEMPLATE)
    endif()

    if(DAWN_COMPILER_GNU)
      dawn_check_and_set_cxx_flag("-fdiagnostics-color=always" HAVE_GCC_COLOR_DIAGNOSTICS)
    endif()
  endif()
endmacro()