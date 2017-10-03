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

#.rst:
# .. _FindLLVM:
#
# FindLLVM
# --------
#
# .. code-block:: cmake
#
#   find_package(LLVM [REQUIRED] [QUIET])
#
# Find the LLVM libraries and includes.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# Defines the following variables:
#
# ``LLVM_FOUND``
#   True if llvm found.
# ``LLVM_INCLUDE_DIRS``
#   Where to find llvm include files
# ``LLVM_LIBRARY_DIRS``
#   Where to find llvm libs
# ``LLVM_CPPFLAGS``
#   LLVM preprocessor flags
# ``LLVM_LDFLAGS``
#   LLVM linker flags.
# ``LLVM_CXXFLAGS`` 
#   LLVM c++ compiler flags
# ``LLVM_INSTALL_PREFIX``
#   LLVM installation prefix
# ``LLVM_VERSION``
#   Version string (``llvm-config --version``)
#
# Hints
# ^^^^^
#
# Set ``LLVM_ROOT`` to a directory that contains a LLVM installation.
#
if (NOT LLVM_ROOT AND DEFINED ENV{LLVM_ROOT})
    file(TO_CMAKE_PATH "$ENV{LLVM_ROOT}" LLVM_ROOT)
endif()

# if the user specified LLVM_ROOT, use that and fail otherwise
if (LLVM_ROOT)
  find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config HINTS ${LLVM_ROOT}/bin DOC "llvm-config executable" NO_DEFAULT_PATH)
else()
  # find llvm-config, prefer the one with a version suffix, e.g. llvm-config-3.5
  # note: FreeBSD installs llvm-config as llvm-config35 and so on
  # note: on some distributions, only 'llvm-config' is shipped, so let's always try to fallback on that
  string(REPLACE "." "" LLVM_FIND_VERSION_CONCAT ${LLVM_FIND_VERSION})
  find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-${LLVM_FIND_VERSION} llvm-config${LLVM_FIND_VERSION_CONCAT} llvm-config DOC "llvm-config executable")

  # other distributions don't ship llvm-config, but only some llvm-config-VERSION binary
  # try to deduce installed LLVM version by looking up llvm-nm in PATH and *then* find llvm-config-VERSION via that
  if (NOT LLVM_CONFIG_EXECUTABLE)
    find_program(_llvmNmExecutable llvm-nm)
    if (_llvmNmExecutable)
      execute_process(COMMAND ${_llvmNmExecutable} --version OUTPUT_VARIABLE _out)
      string(REGEX REPLACE ".*LLVM version ([^ \n]+).*" "\\1" _versionString "${_out}")
      find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-${_versionString} DOC "llvm-config executable")
    endif()
  endif()
endif()

set(LLVM_FOUND FALSE)

if (LLVM_CONFIG_EXECUTABLE)
  # verify that we've found the correct version of llvm-config
  execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --version
    OUTPUT_VARIABLE LLVM_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (NOT LLVM_VERSION)
    set(_LLVM_ERROR_MESSAGE "Failed to parse version from llvm-config")
  elseif (LLVM_FIND_VERSION VERSION_GREATER LLVM_VERSION)
    set(_LLVM_ERROR_MESSAGE "LLVM version too old: ${LLVM_VERSION}")
  else()
    set(LLVM_FOUND TRUE)
  endif()
else()
  set(_LLVM_ERROR_MESSAGE "Could NOT find 'llvm-config' executable")
endif()

if (LLVM_FOUND)
  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir
    OUTPUT_VARIABLE LLVM_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir
    OUTPUT_VARIABLE LLVM_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --cppflags
    OUTPUT_VARIABLE LLVM_CPPFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --ldflags
    OUTPUT_VARIABLE LLVM_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags
    OUTPUT_VARIABLE LLVM_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --libfiles
    OUTPUT_VARIABLE _llvmLibs
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REPLACE " " ";" _llvmLibs ${_llvmLibs})
  foreach(lib ${_llvmLibs})
    list(APPEND LLVM_LIBS "${lib}")
  endforeach()

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs
    OUTPUT_VARIABLE _llvmSystemLibs
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REPLACE " " ";" _llvmSystemLibs ${_llvmSystemLibs})
  foreach(lib ${_llvmSystemLibs})
    list(APPEND LLVM_SYSTEM_LIBS "${lib}")
  endforeach()

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --prefix
    OUTPUT_VARIABLE LLVM_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # potentially add include dir from binary dir for non-installed LLVM
  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --src-root
    OUTPUT_VARIABLE _llvmSourceRoot
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(FIND "${LLVM_INCLUDE_DIRS}" "${_llvmSourceRoot}" _llvmIsInstalled)
  if (NOT _llvmIsInstalled)
    list(APPEND LLVM_INCLUDE_DIRS "${LLVM_INSTALL_PREFIX}/include")
  endif()
endif()

if (LLVM_FIND_REQUIRED AND NOT LLVM_FOUND)
  message(FATAL_ERROR "Could not find LLVM: ${_LLVM_ERROR_MESSAGE}")
elseif(_LLVM_ERROR_MESSAGE)
  message(STATUS "Could not find LLVM: ${_LLVM_ERROR_MESSAGE}")
endif()

if (LLVM_FOUND)
  message(STATUS "Found LLVM (version: ${LLVM_VERSION}): (using ${LLVM_CONFIG_EXECUTABLE})")
  message(STATUS "  Include dirs: ${LLVM_INCLUDE_DIRS}")
  message(STATUS "  Library dirs: ${LLVM_LIBRARY_DIRS}")
endif()

