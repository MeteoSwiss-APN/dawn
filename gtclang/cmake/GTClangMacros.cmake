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

include(CMakePackageConfigHelpers)
include(yodaCheckAndSetCXXFlag)

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

  # Architecture
  yoda_check_and_set_cxx_flag("-march=native" HAVE_GCC_MARCH_NATIVE)
  
  # Pthread
  yoda_check_and_set_cxx_flag("-pthread" HAVE_GCC_PTHREAD)

  # Warnings
  yoda_check_and_set_cxx_flag("-Wall" HAVE_GCC_WALL)
  yoda_check_and_set_cxx_flag("-W" HAVE_GCC_W)
  yoda_check_and_set_cxx_flag("-Wformat" HAVE_GCC_WFORMAT)
  yoda_check_and_set_cxx_flag("-Wmissing-declarations" HAVE_GCC_WMISSING_DECLARATIONS)
  yoda_check_and_set_cxx_flag("-Wmissing-prototypes" HAVE_GCC_WMISSING_PROTOTYPES)
  yoda_check_and_set_cxx_flag("-Wstrict-prototypes" HAVE_GCC_WSTRICT_PROTOTYPES)
  yoda_check_and_set_cxx_flag("-Wundef" HAVE_GCC_WUNDEF)
  yoda_check_and_set_cxx_flag("-Winit-self" HAVE_GCC_WINI_SELF)
  yoda_check_and_set_cxx_flag("-Wpointer-arith" HAVE_GCC_WPOINTER_ARITH)
  yoda_check_and_set_cxx_flag("-Wsign-promo" HAVE_GCC_WSIGN_PROMO)
  yoda_check_and_set_cxx_flag("-Werror=return-type" HAVE_GCC_WERROR_RETURN_TYPE)
  yoda_check_and_set_cxx_flag("-Werror=non-virtual-dtor" HAVE_GCC_WERROR_NON_VIRTUAL_DTOR)
  yoda_check_and_set_cxx_flag("-Werror=address" HAVE_GCC_WERROR_ADDRESS)
  yoda_check_and_set_cxx_flag("-Werror=sequence-point" HAVE_GCC_WERROR_SEQUENCE_POINT)

  yoda_check_and_set_cxx_flag("-Wno-sign-promo" HAVE_GCC_WNO_SIGN_PROMO)
  yoda_check_and_set_cxx_flag("-Wno-sign-compare" HAVE_GCC_WNO_SIGN_COMPARE)
  yoda_check_and_set_cxx_flag("-Wno-unused-parameter" HAVE_GCC_WNO_UNUSDED_PARAMETER)
  yoda_check_and_set_cxx_flag("-Wno-shadow" HAVE_GCC_WNO_SHADOW)
  
  if(BUILD_SHARED_LIBS)
    yoda_check_and_set_cxx_flag("-fPIC" HAVE_GCC_PIC)
  endif()

  if(GTCLANG_USE_CCACHE)
    if(YODA_COMPILER_CLANG)
      yoda_check_and_set_cxx_flag("-Qunused-arguments" HAVE_CLANG_UNUSED_ARGUMENTS)
      yoda_check_and_set_cxx_flag("-fcolor-diagnostics" HAVE_CLANG_COLOR_DIAGNOSTICS)
      yoda_check_and_set_cxx_flag("-Wno-undefined-var-template" 
                                     HAVE_CLANG_WNO_UNDEFINED_VAR_TEMPLATE)
    endif()

    if(YODA_COMPILER_GNU)
      yoda_check_and_set_cxx_flag("-fdiagnostics-color=always" HAVE_GCC_COLOR_DIAGNOSTICS)
    endif()
  endif()
endmacro()

# gtclang_gen_install_config
# -----------------------
#
# Create the CMake packge configuration for installation.
#
macro(gtclang_gen_install_config)
  # Export version
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/gtclangConfigVersion.cmake"
    VERSION ${GTCLANG_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  set(GTCLANG_INSTALL_ROOT "")

  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/templates/gtclangConfig.cmake.in 
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/gtclangConfig.cmake
    INSTALL_DESTINATION ${GTCLANG_INSTALL_CMAKE_DIR}
    PATH_VARS 
      GTCLANG_INSTALL_ROOT
      GTCLANG_INSTALL_INCLUDE_DIR
      GTCLANG_INSTALL_LIB_DIR
      GTCLANG_INSTALL_CMAKE_DIR
      GTCLANG_INSTALL_BIN_DIR
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/gtclangConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/gtclangConfig.cmake"
    DESTINATION ${GTCLANG_INSTALL_CMAKE_DIR}
  )
endmacro()


