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

include(msbuildCheckAndSetCXXFlag)
include(CMakePackageConfigHelpers)

# dawn_set_cxx_flags
# ------------------
#
# Set project specific C++ flags.
#
macro(dawn_set_cxx_flags)
  # Remove -DNDEBUG flag if ASSERTS are ON
  if(DAWN_ASSERTS)
    set(ndebug_flag "-DNDEBUG")
    string(REPLACE "${ndebug_flag}" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "${ndebug_flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  endif()

  # Architecture
  msbuild_check_and_set_cxx_flag("-march=native" HAVE_GCC_MARCH_NATIVE)
  
  # Pthread
  msbuild_check_and_set_cxx_flag("-pthread" HAVE_GCC_PTHREAD)

  # Warnings
  msbuild_check_and_set_cxx_flag("-Wall" HAVE_GCC_WALL)
  msbuild_check_and_set_cxx_flag("-W" HAVE_GCC_W)
  msbuild_check_and_set_cxx_flag("-Wformat" HAVE_GCC_WFORMAT)
  msbuild_check_and_set_cxx_flag("-Wmissing-declarations" HAVE_GCC_WMISSING_DECLARATIONS)
  msbuild_check_and_set_cxx_flag("-Wmissing-prototypes" HAVE_GCC_WMISSING_PROTOTYPES)
  msbuild_check_and_set_cxx_flag("-Wstrict-prototypes" HAVE_GCC_WSTRICT_PROTOTYPES)
  msbuild_check_and_set_cxx_flag("-Wundef" HAVE_GCC_WUNDEF)
  msbuild_check_and_set_cxx_flag("-Winit-self" HAVE_GCC_WINI_SELF)
  msbuild_check_and_set_cxx_flag("-Wpointer-arith" HAVE_GCC_WPOINTER_ARITH)
  msbuild_check_and_set_cxx_flag("-Wshadow" HAVE_GCC_WSHADOW)
  msbuild_check_and_set_cxx_flag("-Wsign-promo" HAVE_GCC_WSIGN_PROMO)
  msbuild_check_and_set_cxx_flag("-Werror=return-type" HAVE_GCC_WERROR_RETURN_TYPE)
  msbuild_check_and_set_cxx_flag("-Werror=non-virtual-dtor" HAVE_GCC_WERROR_NON_VIRTUAL_DTOR)
  msbuild_check_and_set_cxx_flag("-Werror=address" HAVE_GCC_WERROR_ADDRESS)
  msbuild_check_and_set_cxx_flag("-Werror=sequence-point" HAVE_GCC_WERROR_SEQUENCE_POINT)

  msbuild_check_and_set_cxx_flag("-Wno-sign-promo" HAVE_GCC_WNO_SIGN_PROMO)
  msbuild_check_and_set_cxx_flag("-Wno-sign-compare" HAVE_GCC_WNO_SIGN_COMPARE)
  msbuild_check_and_set_cxx_flag("-Wno-unused-parameter" HAVE_GCC_WNO_UNUSDED_PARAMETER)
  
  if(BUILD_SHARED_LIBS)
    msbuild_check_and_set_cxx_flag("-fPIC" HAVE_GCC_PIC)
  endif()

  if(DAWN_USE_CCACHE)
    if(DAWN_COMPILER_CLANG)
      msbuild_check_and_set_cxx_flag("-Qunused-arguments" HAVE_CLANG_UNUSED_ARGUMENTS)
      msbuild_check_and_set_cxx_flag("-fcolor-diagnostics" HAVE_CLANG_COLOR_DIAGNOSTICS)
      msbuild_check_and_set_cxx_flag("-Wno-undefined-var-template" 
                                     HAVE_CLANG_WNO_UNDEFINED_VAR_TEMPLATE)
    endif()

    if(DAWN_COMPILER_GNU)
      msbuild_check_and_set_cxx_flag("-fdiagnostics-color=always" HAVE_GCC_COLOR_DIAGNOSTICS)
    endif()
  endif()
endmacro()

# dawn_gen_install_config
# -----------------------
#
# Create the CMake packge configuration for installation.
#
macro(dawn_gen_install_config)
  # Export version
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DawnConfigVersion.cmake"
    VERSION ${DAWN_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  set(DAWN_INSTALL_ROOT "")

  # Export configuration
  set(DAWN_INSTALL_PROTOBUF_ROOT ${DAWN_PROTOBUF_ROOT})
  set(DAWN_INSTALL_PROTOBUF_VERSION ${DAWN_PROTOBUF_VERSION})

  configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/templates/DawnConfig.cmake.in 
    ${CMAKE_CURRENT_BINARY_DIR}/DawnConfig.cmake
    INSTALL_DESTINATION ${DAWN_INSTALL_CMAKE_DIR}
    PATH_VARS 
      DAWN_INSTALL_ROOT
      DAWN_INSTALL_INCLUDE_DIR
      DAWN_INSTALL_LIB_DIR
      DAWN_INSTALL_CMAKE_DIR
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DawnConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DawnConfig.cmake"
    DESTINATION ${DAWN_INSTALL_CMAKE_DIR}
  )
endmacro()
