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

find_path(Clang_INCLUDE_DIR NAMES clang/AST/AST.h
    HINTS ${CLANG_INCLUDE_DIRS} ${LLVM_ROOT}
    PATH_SUFFIXES include clang
)

if(NOT Clang_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find clang headers: Try setting CLANG_INCLUDE_DIRS or LLVM_ROOT")
endif()

# Generated for Clang 3.8.0 but should still work
set(clang_libs
  clang
  clangARCMigrate
  clangAST
  clangASTMatchers
  clangAnalysis
  clangBasic
  clangCheckers
  clangCodeGen
  clangCore
  clangDriver
  clangDynamic
  clangEdit
  clangFormat
  clangFrontend
  clangFrontendTool
  clangIndex
  clangLex
  clangParse
  clangRewrite
  clangSema
  clangSerialization
  clangTooling
)

add_library(Clang INTERFACE)
foreach(_lib ${clang_libs})
    find_library(library ${_lib}
      HINTS ${CLANG_LIB_DIRS} ${LLVM_ROOT}
      PATH_SUFFIXES lib
    )
    if(library)
      add_library(${_lib} MODULE IMPORTED)
      target_compile_features(${_lib} INTERFACE cxx_std_11)
      target_include_directories(${_lib}
        INTERFACE ${Clang_INCLUDE_DIR}
      )
      target_link_libraries(${_lib} INTERFACE ${library})
      target_link_libraries(Clang INTERFACE ${_lib})
    else()
      message(FATAL_ERROR "Could not find ${_lib} library: Try setting CLANG_LIB_DIRS or LLVM_ROOT")
    endif()
endforeach()

mark_as_advanced(Clang_INCLUDE_DIR)

# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(RapidJSON
#     REQUIRED_VARS RapidJSON_INCLUDE_DIR
#     VERSION_VAR RapidJSON_VERSION
# )
