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

include(yodaCheckAndSetCXXFlag)
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
  yoda_check_and_set_cxx_flag("-Wshadow" HAVE_GCC_WSHADOW)
  yoda_check_and_set_cxx_flag("-Wsign-promo" HAVE_GCC_WSIGN_PROMO)
  yoda_check_and_set_cxx_flag("-Werror=return-type" HAVE_GCC_WERROR_RETURN_TYPE)
  yoda_check_and_set_cxx_flag("-Werror=non-virtual-dtor" HAVE_GCC_WERROR_NON_VIRTUAL_DTOR)
  yoda_check_and_set_cxx_flag("-Werror=address" HAVE_GCC_WERROR_ADDRESS)
  yoda_check_and_set_cxx_flag("-Werror=sequence-point" HAVE_GCC_WERROR_SEQUENCE_POINT)

  yoda_check_and_set_cxx_flag("-Wno-sign-promo" HAVE_GCC_WNO_SIGN_PROMO)
  yoda_check_and_set_cxx_flag("-Wno-sign-compare" HAVE_GCC_WNO_SIGN_COMPARE)
  yoda_check_and_set_cxx_flag("-Wno-unused-parameter" HAVE_GCC_WNO_UNUSDED_PARAMETER)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    yoda_check_and_set_cxx_flag("-fsanitize=address" HAVE_GCC_SANITIZE_ADDRESS)
  endif()

  if(BUILD_SHARED_LIBS)
    yoda_check_and_set_cxx_flag("-fPIC" HAVE_GCC_PIC)
  endif()

  if(DAWN_USE_CCACHE)
    if(DAWN_COMPILER_CLANG)
      yoda_check_and_set_cxx_flag("-Qunused-arguments" HAVE_CLANG_UNUSED_ARGUMENTS)
      yoda_check_and_set_cxx_flag("-fcolor-diagnostics" HAVE_CLANG_COLOR_DIAGNOSTICS)
      yoda_check_and_set_cxx_flag("-Wno-undefined-var-template"
                                     HAVE_CLANG_WNO_UNDEFINED_VAR_TEMPLATE)
    endif()

    if(DAWN_COMPILER_GNU)
      yoda_check_and_set_cxx_flag("-fdiagnostics-color=always" HAVE_GCC_COLOR_DIAGNOSTICS)
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

  set(DAWN_CONFIGURE_ROOT "")
  set(DAWN_CONFIGURE_CMAKE_DIR "${DAWN_INSTALL_CMAKE_DIR}")
  set(DAWN_CONFIGURE_INCLUDE_DIR "${DAWN_INSTALL_INCLUDE_DIR}")
  set(DAWN_CONFIGURE_EXTERNAL_INCLUDE_DIR "${DAWN_CONFIGURE_INCLUDE_DIR}/dawn/Support/External")
  set(DAWN_CONFIGURE_GENERATED_INCLUDE_DIR "${DAWN_INSTALL_INCLUDE_DIR}")
  set(DAWN_CONFIGURE_LIB_DIR "${DAWN_INSTALL_LIB_DIR}")

  # Export configuration
  set(DAWN_CONFIGURE_PROTOBUF_ROOT ${DAWN_PROTOBUF_ROOT})
  set(DAWN_CONFIGURE_PROTOBUF_VERSION ${DAWN_PROTOBUF_VERSION})

  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/dawnConfig.cmake.in
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/dawnConfig.cmake
    INSTALL_DESTINATION ${DAWN_INSTALL_CMAKE_DIR}
    PATH_VARS
      DAWN_CONFIGURE_ROOT
      DAWN_CONFIGURE_INCLUDE_DIR
      DAWN_CONFIGURE_EXTERNAL_INCLUDE_DIR
      DAWN_CONFIGURE_GENERATED_INCLUDE_DIR
      DAWN_CONFIGURE_LIB_DIR
      DAWN_CONFIGURE_CMAKE_DIR
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dawnConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/dawnConfig.cmake"
    DESTINATION ${DAWN_INSTALL_CMAKE_DIR}
  )

  # in source installation
  set(DAWN_CONFIGURE_ROOT "${PROJECT_BINARY_DIR}")
  set(DAWN_CONFIGURE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src")
  set(DAWN_CONFIGURE_EXTERNAL_INCLUDE_DIR "${DAWN_CONFIGURE_INCLUDE_DIR}/dawn/Support/External")
  set(DAWN_CONFIGURE_GENERATED_INCLUDE_DIR "${PROJECT_BINARY_DIR}/src")
  set(DAWN_CONFIGURE_LIB_DIR "")
  set(DAWN_CONFIGURE_CMAKE_DIR "${PROJECT_BINARY_DIR}")
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/dawnConfig.cmake.in
    ${PROJECT_BINARY_DIR}/dawnConfig.cmake
    INSTALL_DESTINATION ${PROJECT_BINARY_DIR}
    PATH_VARS
      DAWN_CONFIGURE_ROOT
      DAWN_CONFIGURE_INCLUDE_DIR
      DAWN_CONFIGURE_EXTERNAL_INCLUDE_DIR
      DAWN_CONFIGURE_GENERATED_INCLUDE_DIR
      DAWN_CONFIGURE_LIB_DIR
      DAWN_CONFIGURE_CMAKE_DIR
  )

endmacro()
