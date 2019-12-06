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

set(config_names 6.0 7.0 8.0 9.0 10.0)
list(TRANSFORM config_names PREPEND "llvm-config-")
list(APPEND config_names "llvm-config")

# if the user specified LLVM_ROOT, use that and fail otherwise
if (LLVM_ROOT)
  find_program(LLVM_CONFIG_EXECUTABLE
    NAMES ${_config_names}
    HINTS ${LLVM_ROOT}/bin
    DOC "llvm-config executable"
    NO_DEFAULT_PATH
  )
  mark_as_advanced(LLVM_CONFIG_EXECUTABLE)
else()
  # find llvm-config, prefer the one with a version suffix, e.g. llvm-config-3.5
  # note: FreeBSD installs llvm-config as llvm-config35 and so on
  # note: on some distributions, only 'llvm-config' is shipped, so let's always try to fallback on that
  string(REPLACE "." "" LLVM_FIND_VERSION_CONCAT "${LLVM_FIND_VERSION}")
  find_program(LLVM_CONFIG_EXECUTABLE
    NAMES llvm-config-${LLVM_FIND_VERSION} llvm-config${LLVM_FIND_VERSION_CONCAT} llvm-config
    DOC "llvm-config executable"
  )

  # other distributions don't ship llvm-config, but only some llvm-config-VERSION binary
  # try to deduce installed LLVM version by looking up llvm-nm in PATH and *then* find llvm-config-VERSION via that
  if (NOT LLVM_CONFIG_EXECUTABLE)
    find_program(_llvmNmExecutable llvm-nm)
    if (_llvmNmExecutable)
      execute_process(COMMAND ${_llvmNmExecutable} --version OUTPUT_VARIABLE _out)
      string(REGEX REPLACE ".*LLVM version ([^ \n]+).*" "\\1" _versionString "${_out}")
      find_program(LLVM_CONFIG_EXECUTABLE
        NAMES llvm-config-${_versionString}
        DOC "llvm-config executable"
      )
    endif()
  endif()
endif()

set(LLVM_FOUND FALSE)

if (LLVM_CONFIG_EXECUTABLE)
  # verify that we've found the correct version of llvm-config
  execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --version
    OUTPUT_VARIABLE LLVM_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

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

macro(get_llvm_variable_as_list flag var)
  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} ${flag}
    OUTPUT_VARIABLE _out
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REPLACE " " ";" _out "${_out}")
  foreach(token ${_out})
    list(APPEND ${var} ${token})
  endforeach()
endmacro()

if (LLVM_FOUND)
  get_llvm_variable_as_list(--includedir LLVM_INCLUDE_DIRS)
  get_llvm_variable_as_list(--libdir LLVM_LIBRARY_DIRS)
  get_llvm_variable_as_list(--cxxflags LLVM_CXXFLAGS)
  get_llvm_variable_as_list(--libfiles LLVM_LIBS)
  get_llvm_variable_as_list(--system-libs LLVM_SYSTEM_LIBS)
  get_llvm_variable_as_list(--src-root llvm_source_root)
  get_llvm_variable_as_list(--has-rtti has_rtti)
  if (has_rtti STREQUAL NO)
    message(FATAL_ERROR "GTClang requires LLVM built with RTTI")
  endif()
endif()

if (LLVM_FIND_REQUIRED AND NOT LLVM_FOUND)
  message(FATAL_ERROR "Could not find LLVM: ${_LLVM_ERROR_MESSAGE}")
elseif(_LLVM_ERROR_MESSAGE)
  message(STATUS "Could not find LLVM: ${_LLVM_ERROR_MESSAGE}")
endif()

set(FILTERED_LLVM_CXXFLAGS)
foreach(flag ${LLVM_CXXFLAGS})
  # Filter definitions
  if(${flag} MATCHES "^-D.*$" AND NOT(${flag} STREQUAL "-DNDEBUG"))
    list(APPEND FILTERED_LLVM_CXXFLAGS ${flag})
    continue()
  endif()

  # Filter includes
  if(${flag} MATCHES "^-I.*$")
    list(APPEND FILTERED_LLVM_CXXFLAGS ${flag})
    continue()
  endif()

  # Filter -f* options (we want exceptions tough!)
  if(${flag} MATCHES "^-f.*$" AND NOT(${flag} STREQUAL "-fno-exceptions"))
    list(APPEND FILTERED_LLVM_CXXFLAGS ${flag})
  endif()
endforeach()

add_library(LLVM INTERFACE IMPORTED GLOBAL)
add_library(LLVM::LLVM ALIAS LLVM)
target_include_directories(LLVM INTERFACE ${LLVM_INCLUDE_DIRS})
target_link_libraries(LLVM INTERFACE ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS})
target_compile_options(LLVM INTERFACE ${FILTERED_LLVM_CXXFLAGS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LLVM
  REQUIRED_VARS LLVM_CONFIG_EXECUTABLE LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIRS LLVM_LIBS
  VERSION_VAR LLVM_VERSION
)
