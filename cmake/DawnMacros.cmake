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

include(mchbuildCheckAndSetCXXFlag)
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
  mchbuild_check_and_set_cxx_flag("-march=native" HAVE_GCC_MARCH_NATIVE)
  
  # Pthread
  mchbuild_check_and_set_cxx_flag("-pthread" HAVE_GCC_PTHREAD)

  # Warnings
  mchbuild_check_and_set_cxx_flag("-Wall" HAVE_GCC_WALL)
  mchbuild_check_and_set_cxx_flag("-W" HAVE_GCC_W)
  mchbuild_check_and_set_cxx_flag("-Wformat" HAVE_GCC_WFORMAT)
  mchbuild_check_and_set_cxx_flag("-Wmissing-declarations" HAVE_GCC_WMISSING_DECLARATIONS)
  mchbuild_check_and_set_cxx_flag("-Wmissing-prototypes" HAVE_GCC_WMISSING_PROTOTYPES)
  mchbuild_check_and_set_cxx_flag("-Wstrict-prototypes" HAVE_GCC_WSTRICT_PROTOTYPES)
  mchbuild_check_and_set_cxx_flag("-Wundef" HAVE_GCC_WUNDEF)
  mchbuild_check_and_set_cxx_flag("-Winit-self" HAVE_GCC_WINI_SELF)
  mchbuild_check_and_set_cxx_flag("-Wpointer-arith" HAVE_GCC_WPOINTER_ARITH)
  mchbuild_check_and_set_cxx_flag("-Wshadow" HAVE_GCC_WSHADOW)
  mchbuild_check_and_set_cxx_flag("-Wsign-promo" HAVE_GCC_WSIGN_PROMO)
  mchbuild_check_and_set_cxx_flag("-Werror=return-type" HAVE_GCC_WERROR_RETURN_TYPE)
  mchbuild_check_and_set_cxx_flag("-Werror=non-virtual-dtor" HAVE_GCC_WERROR_NON_VIRTUAL_DTOR)
  mchbuild_check_and_set_cxx_flag("-Werror=address" HAVE_GCC_WERROR_ADDRESS)
  mchbuild_check_and_set_cxx_flag("-Werror=sequence-point" HAVE_GCC_WERROR_SEQUENCE_POINT)

  mchbuild_check_and_set_cxx_flag("-Wno-sign-promo" HAVE_GCC_WNO_SIGN_PROMO)
  mchbuild_check_and_set_cxx_flag("-Wno-sign-compare" HAVE_GCC_WNO_SIGN_COMPARE)
  mchbuild_check_and_set_cxx_flag("-Wno-unused-parameter" HAVE_GCC_WNO_UNUSDED_PARAMETER)
  
  if(BUILD_SHARED_LIBS)
    mchbuild_check_and_set_cxx_flag("-fPIC" HAVE_GCC_PIC)
  endif()

  if(DAWN_USE_CCACHE)
    if(DAWN_COMPILER_CLANG)
      mchbuild_check_and_set_cxx_flag("-Qunused-arguments" HAVE_CLANG_UNUSED_ARGUMENTS)
      mchbuild_check_and_set_cxx_flag("-fcolor-diagnostics" HAVE_CLANG_COLOR_DIAGNOSTICS)
      mchbuild_check_and_set_cxx_flag("-Wno-undefined-var-template" 
                                     HAVE_CLANG_WNO_UNDEFINED_VAR_TEMPLATE)
    endif()

    if(DAWN_COMPILER_GNU)
      mchbuild_check_and_set_cxx_flag("-fdiagnostics-color=always" HAVE_GCC_COLOR_DIAGNOSTICS)
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
    "${CMAKE_CURRENT_BINARY_DIR}/dawnConfigVersion.cmake"
    VERSION ${DAWN_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  set(DAWN_INSTALL_ROOT "")

  # Export configuration
  set(DAWN_INSTALL_PROTOBUF_ROOT ${DAWN_PROTOBUF_ROOT})
  set(DAWN_INSTALL_PROTOBUF_VERSION ${DAWN_PROTOBUF_VERSION})

  configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/templates/dawnConfig.cmake.in 
    ${CMAKE_CURRENT_BINARY_DIR}/dawnConfig.cmake
    INSTALL_DESTINATION ${DAWN_INSTALL_CMAKE_DIR}
    PATH_VARS 
      DAWN_INSTALL_ROOT
      DAWN_INSTALL_INCLUDE_DIR
      DAWN_INSTALL_LIB_DIR
      DAWN_INSTALL_CMAKE_DIR
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dawnConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dawnConfig.cmake"
    DESTINATION ${DAWN_INSTALL_CMAKE_DIR}
  )
endmacro()
