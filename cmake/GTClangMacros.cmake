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

gtclang_include_guard()

include(CheckCXXCompilerFlag)

## gtclang_check_in_source_build
## -----------------------------
##
## Prevent in source builds.
##
macro(gtclang_check_in_source_build)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
    message(FATAL_ERROR "In-source builds are not allowed.
Please create a build directory and run cmake from there, passing the path
to this source directory as the last argument.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
Please delete them.")
  endif()
endmacro()

## gtclang_export_package_variable
## -------------------------------
##
## Export the variables:
##  GTCLANG_${PACKAGE}_FOUND      : Package was found
##  GTCLANG_${PACKAGE}_INFO_STR   : Information about the package (e.g version string)
##  GTCLANG_${PACKAGE}_LIBRARIES  : Libraries of the package to link against
##
## where
##
##    PACKAGE:STRING=<>    - Name of the package
##    FOUND:BOOL=<>        - Package was found
##    INFO_STR:STRING=<>   - Information about the package (e.g version string)
##    ARGN                 - List of libraries to link against
##
macro(gtclang_export_package_variable PACKAGE FOUND INFO_STR)
  set("GTCLANG_${PACKAGE}_FOUND" ${FOUND} CACHE BOOL "${PACKAGE} found" FORCE)
  set("GTCLANG_${PACKAGE}_INFO_STR" ${INFO_STR} CACHE STRING "Info of package: ${PACKAGE}" FORCE)
  set("GTCLANG_${PACKAGE}_LIBRARIES" ${ARGN} CACHE STRING "Libraries of package: ${PACKAGE}" FORCE)
  mark_as_advanced("GTCLANG_${PACKAGE}_FOUND" 
                   "GTCLANG_${PACKAGE}_INFO_STR" 
                   "GTCLANG_${PACKAGE}_LIBRARIES")
endmacro()

## gtclang_check_and_set_cxx_flag
## ------------------------------
##
## Test if the compiler flag exists and add it to the ``CMAKE_CXX_FLAGS``.
##
##    FLAG:STRING=<>   - Compiler flag to check (e.g -O3)
##    NAME:STRING=<>   - Name of the check (e.g HAVE_GCC_O3)
##
macro(gtclang_check_and_set_cxx_flag FLAG NAME)
  check_cxx_compiler_flag("${FLAG}" ${NAME})
  if(${NAME})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
  endif()
endmacro()

## gtclang_set_cxx_standard
## ------------------------
##
## Set the standard of C++.
##
##    CXX_STANDARD:STRING=<>   - C++ standard to use, one of ["c++03", "c++11", "c++14"]
##
macro(gtclang_set_cxx_standard CXX_STANDARD)
    set(supported_standards "c++03" "c++11" "c++14")
    
    if(NOT("${CXX_STANDARD}" IN_LIST supported_standards))
      message(FATAL_ERROR 
              "Unknown C++ standard (${CXX_STANDARD}), supported values are: ${supported_standards}")
    endif()

    set(cxx_stanard_flag "-std=${CXX_STANDARD}")

    check_cxx_compiler_flag("${cxx_stanard_flag}" HAVE_CXX_STANDARD_SUPPORT)
    
    if(NOT(HAVE_CXX_STANDARD_SUPPORT))
      message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no ${CXX_STANDARD} support.")
    endif()
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx_stanard_flag}")
endmacro()

## gtclang_set_cxx_flags
## ---------------------
##
## Set C++ flags.
##
macro(gtclang_set_cxx_flags)

  # Remove -DNDEBUG flag if ASSERTS are ON
  if(GTCLANG_ASSERTS)
    set(ndebug_flag "-DNDEBUG")
    string(REPLACE "${ndebug_flag}" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "${ndebug_flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  endif()
  
  gtclang_check_and_set_cxx_flag("-march=native" HAVE_GCC_MARCH_NATIVE)
  gtclang_check_and_set_cxx_flag("-Wall" HAVE_GCC_WALL)
  gtclang_check_and_set_cxx_flag("-Wno-sign-compare" HAVE_GCC_WNO_SIGN_COMPARE)
  gtclang_check_and_set_cxx_flag("-Werror=return-type" HAVE_GCC_ERROR_RETURN_TYPE)
endmacro()

## gtclang_check_compiler
## ----------------------
##
## Check whether the current compiler is supported, and provide friendly hints to the user.
##
macro(gtclang_check_compiler)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "3.4")
          message(WARNING "
      ### You appear to be using Clang ${CMAKE_CXX_COMPILER_VERSION}, which is known
      ### to be unable to compile GTCLANG. Consider switching to
      ### Clang >= 3.5. If it is already installed on your
      ### system, you can tell CMake about it with
      ###
      ###     cmake .. -DCMAKE_CXX_COMPILER=/path/to/clang
          ")
      endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
      if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "6.1.0")
          message(WARNING "
      ### You appear to be using Apple's Clang ${CMAKE_CXX_COMPILER_VERSION}, which is
      ### shipped with Xcode < 6.3. Unfortunately, only Apple's Clang
      ### >= 6.1.0 (shipped with Xcode >= 6.3) is supported by GTCLANG.
      ### You should consider updating to Xcode >= 6.3 (requires Yosemite)
      ### or using a non-Apple Clang >= 3.4, which can be installed via
      ### Homebrew with
      ###
      ###     brew install llvm --with-clang
      ###
      ### You can then tell CMake to use that non-system Clang with
      ###
      ###     cmake .. -DCMAKE_CXX_COMPILER=/path/to/clang
          ")
      endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "17")
          message(WARNING "
      ### You appear to be using Intel's ICC ${CMAKE_CXX_COMPILER_VERSION}, which is known
      ### to be unable to compile GTCLANG. Consider switching to
      ### ICC >= 17.0. If it is already installed on your
      ### system, you can tell CMake about it with
      ###
      ###     cmake .. -DCMAKE_CXX_COMPILER=/path/to/icpc
          ")
      endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "4.9")
          message(WARNING "
      ### You appear to be using GCC ${CMAKE_CXX_COMPILER_VERSION}, which is known to be
      ### unable to compile GTCLANG. Only GCC >= 4.9 is supported.
      ### Consider using a more recent GCC or switching to Clang.
      ### If a more recent compiler is already installed on your
      ### system, you can tell CMake to use it with
      ###
      ###     cmake .. -DCMAKE_CXX_COMPILER=/path/to/g++
          ")
      endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "19.0")
          message(WARNING "
      ### You appear to be using Visual Studio ${CMAKE_CXX_COMPILER_VERSION}, 
      ### which is known to be unable to compile GTCLANG. 
      ### Only Visual Studio 14 2015 is currently supported.
          ")
      endif()
  else()
      message(WARNING "
      ### You appear to be using a compiler that is not yet tested with GTCLANG.
      ")
  endif()
endmacro()

## gtclang_add_target_clean_all
## ----------------------------
##
## Provide a `clean-all` target which clears the CMake Cache.
##
macro(gtclang_add_target_clean_all)
  add_custom_target(clean-all
          COMMAND ${CMAKE_MAKE_PROGRAM} clean
          COMMAND ${CMAKE_COMMAND} -P 
                  "${CMAKE_SOURCE_DIR}/cmake/scripts/CleanAll.cmake")
endmacro()

## gtclang_make_build_script
## -------------------------
##
## Provide a bash script to build `gtclang` and `gsl` together.
##
macro(gtclang_make_build_script)
  set(GSL_BUILD_COMMAND "${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/../gsl --target install")
  set(GTCLANG_BUILD_COMMAND "${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all")

  if(${CMAKE_GENERATOR} STREQUAL "Unix Makefiles")
    set(GSL_BUILD_COMMAND "${GSL_BUILD_COMMAND} -- -j4")
    set(GTCLANG_BUILD_COMMAND "${GTCLANG_BUILD_COMMAND} -- -j4")
  endif()

  file(WRITE ${CMAKE_BINARY_DIR}/build.sh "#!/bin/bash\n")
  file(APPEND ${CMAKE_BINARY_DIR}/build.sh "${GSL_BUILD_COMMAND}\n")
  file(APPEND ${CMAKE_BINARY_DIR}/build.sh "${GTCLANG_BUILD_COMMAND}\n")
endmacro()

## gtclang_add_target_clang_format
## -------------------------------
##
## Setup a target for clang-format:
##    check-format      Runs clang format and exits with a non-zero exit code if any files need to 
##                      be reformatted
##    format            Runs clang format and updates files in place
##
## If clang-format cannot be found, try setting CLANG_FORMAT_BIN to the clang-format binary.
## The targets are only available on Unix like systems.
##
##    SOURCE_DIR:STRING=<>      - Run clang-format recursivly on all files in SOURCE_DIR with "*h"
##                                and ".cpp" extensions. 
##
macro(gtclang_add_target_clang_format SOURCE_DIR) 
  
  find_program(CLANG_FORMAT_BIN 
    NAMES clang-format-3.8 clang-format-3.7 clang-format-3.6 clang-format
    PATHS ${CLANG_FORMAT_PATH} $ENV{CLANG_FORMAT_PATH} /usr/local/bin /usr/bin 
          NO_DEFAULT_PATH
  )

  set(source_dir ${CMAKE_SOURCE_DIR}/src)

  if(NOT("${CLANG_FORMAT_BIN}" STREQUAL "CLANG_FORMAT_BIN-NOTFOUND"))
    # Runs clang format and updates files in place.
    add_custom_target(format 
                      COMMAND bash ${PROJECT_SOURCE_DIR}/cmake/scripts/run-clang-format.sh 
                      ${CMAKE_SOURCE_DIR} 
                      ${CLANG_FORMAT_BIN} 
                      1
                      `find ${source_dir} -name \*.h -print -o -name \*.cpp -print -o -name \*.hpp -print`)

    # Runs clang format and exits with a non-zero exit code if any files need to be reformatted
    add_custom_target(check-format 
                      COMMAND bash ${PROJECT_SOURCE_DIR}/cmake/scripts/run-clang-format.sh 
                      ${CMAKE_SOURCE_DIR} 
                      ${CLANG_FORMAT_BIN} 
                      0
                      `find ${source_dir} -name \*.h -print -o -name \*.cpp -print -o -name \*.hpp -print`)
  endif()
endmacro()


## gtclang_set_install_dirs
## ------------------------
##
## Some default installation locations. These should be global, with any project specific locations 
## added to the end. These paths are all relative to the install prefix.
##
## These paths attempt to adhere to the FHS, and are similar to those provided by autotools and used 
## in many Linux distributions.
##
macro(gtclang_set_install_dirs)

  # Use GNU install directories
  include(GNUInstallDirs)
  if(NOT INSTALL_RUNTIME_DIR)
    set(INSTALL_RUNTIME_DIR "${CMAKE_INSTALL_BINDIR}")
  endif()
  if(NOT INSTALL_LIBRARY_DIR)
    set(INSTALL_LIBRARY_DIR "${CMAKE_INSTALL_LIBDIR}")
  endif()
  if(NOT INSTALL_ARCHIVE_DIR)
    set(INSTALL_ARCHIVE_DIR "${CMAKE_INSTALL_LIBDIR}")
  endif()
  if(NOT INSTALL_INCLUDE_DIR)
    set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
  endif()
  if(NOT INSTALL_DATA_DIR)
    set(INSTALL_DATA_DIR "${CMAKE_INSTALL_DATAROOTDIR}")
  endif()
  if(NOT INSTALL_DOC_DIR)
    set(INSTALL_DOC_DIR "${CMAKE_INSTALL_DOCDIR}")
  endif()
  if(NOT INSTALL_MAN_DIR)
    set(INSTALL_MAN_DIR "${CMAKE_INSTALL_MANDIR}")
  endif()
  if(UNIX AND NOT APPLE)
    if(NOT INSTALL_XDG_APP_DIR)
      set(INSTALL_XDG_APPS_DIR "${INSTALL_DATA_DIR}/applications")
    endif()
    if(NOT INSTALL_XDG_ICON_DIR)
      set(INSTALL_XDG_ICON_DIR "${INSTALL_DATA_DIR}/pixmaps")
    endif()
  endif()

  # Set up RPATH for the project too.
  if(NOT CMAKE_INSTALL_RPATH)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${INSTALL_LIBRARY_DIR}")
  endif()
  if(APPLE AND NOT CMAKE_INSTALL_NAME_DIR)
    set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${INSTALL_LIBRARY_DIR}")
  endif()
  if(UNIX AND GTCLANG_ENABLE_RPATH)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif()
endmacro()
