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

include(yodaCheckAndSetCXXFlag)
include(yodaExportPackage)

set(llvm_clang_version "3.8.0")
set(llvm_clang_version_short "3.8")

find_package(Clang ${llvm_clang_version_short} REQUIRED)

list(APPEND clang_libraries ${CLANG_LIBS} ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS})
list(APPEND clang_include_dirs ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
set(clang_definitions)

# Set the resource path (this path contains the LLVM/Clang specific include files like `stdarg.h` 
# which are needed during the invocation of clang)
set(GTCLANG_CLANG_RESSOURCE_INCLUDE_PATH ${CLANG_RESSOURCE_INCLUDE_PATH})

# Parse the C++ flags of LLVM/Clang and properly convert them to CMake
string(REPLACE " " ";" llvm_cxx_falgs ${LLVM_CXXFLAGS})
foreach(flag ${llvm_cxx_falgs})

  # Filter definitions
  if(${flag} MATCHES "^-D.*$" AND NOT(${flag} STREQUAL "-DNDEBUG"))
    list(APPEND clang_definitions ${flag})
    continue()
  endif()

  # Filter includes
  if(${flag} MATCHES "^-I.*$")
    string(REPLACE "-I" "" include_path ${flag})
    list(APPEND clang_include_dirs ${include_path})
    continue()
  endif()

  # Filter -f* options (we want exceptions tough!)
  if(${flag} MATCHES "^-f.*$" AND NOT(${flag} STREQUAL "-fno-exceptions"))
    string(REGEX REPLACE "-f" "" check_name ${flag})
    string(REGEX REPLACE "-" "_" check_name ${check_name})
    string(TOUPPER ${check_name} check_name)
    set(check_name "HAVE_${check_name}")
    yoda_check_and_set_cxx_flag("${flag}" ${check_name})
    continue()
  endif()
endforeach()

yoda_export_package(
  NAME Clang
  FOUND ${CLANG_FOUND}
  VERSION "${LLVM_VERSION}" 
  LIBRARIES ${clang_libraries}
  INCLUDE_DIRS ${clang_include_dirs}
  DEFINITIONS ${clang_definitions}
)
