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

if(NOT TARGET LLVM::LLVM)
  if(${Clang_FIND_REQUIRED})
    find_package(LLVM ${Clang_FIND_VERSION} REQUIRED)
  else()
    find_package(LLVM ${Clang_FIND_VERSION})
  endif()
endif()

set(CLANG_FOUND FALSE)

if(LLVM_FOUND)
  set(CLANG_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
  set(CLANG_LIBRARY_DIRS ${LLVM_LIBRARY_DIRS})
  set(CLANG_VERSION ${LLVM_VERSION})

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --prefix
    OUTPUT_VARIABLE llvm_install_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
set(CLANG_RESSOURCE_INCLUDE_PATH "${llvm_install_prefix}/lib/clang/${LLVM_VERSION}/include")

  # potentially add include dir from binary dir for non-installed LLVM
  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --src-root
    OUTPUT_VARIABLE llvm_source_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(FIND "${LLVM_INCLUDE_DIRS}" "${llvm_source_root}" str_found)
  list(LENGTH LLVM_INCLUDE_DIRS num_include_dirs)
  if((num_include_dirs EQUAL 1 AND (llvm_source_root STREQUAL LLVM_INCLUDE_DIRS)) OR
      (num_include_dirs GREATER 1 AND str_found GREATER_EQUAL 0))
    message(STATUS
      "Detected that llvm-config comes from a build-tree, adding more include directories for Clang")
    list(APPEND CLANG_INCLUDE_DIRS
         "${llvm_install_prefix}/tools/clang/include" # build dir
         "${llvm_source_root}/tools/clang/include"     # source dir
    )
  endif()

  # TODO test this set with all supported LLVM versions
  # 6: done
  # 7: done
  # 8: done
  # 9
  # 10 (master):
  set(clang_libnames
    clangFrontend
    clangARCMigrate
    clangASTMatchers
    clangAnalysis
    clangCodeGen
    clangDriver
    clangEdit
    clangFormat
    clangFrontendTool
    clangIndex
    clangParse
    clangRewrite
    clangRewriteFrontend
    clangSema
    clangAnalysis
    clangSerialization
    clangStaticAnalyzerCheckers
    clangStaticAnalyzerCore
    clangTooling
    clangToolingCore
    clangAST
    clangLex
    clangBasic
  )
  if(${LLVM_VERSION} VERSION_GREATER_EQUAL 7.0.0)
    list(APPEND clang_libnames clangToolingInclusions)
  endif()

  add_library(Clang INTERFACE IMPORTED GLOBAL)
  add_library(Clang::Clang ALIAS Clang)
  target_include_directories(Clang INTERFACE ${CLANG_INCLUDE_DIRS})
  target_compile_features(Clang INTERFACE cxx_std_11)
  foreach(_lib ${clang_libnames})
    find_library(${_lib}_LIBRARY ${_lib}
      HINTS ${CLANG_LIBRARY_DIRS} ${LLVM_ROOT}
      PATH_SUFFIXES lib
      )
    mark_as_advanced(${_lib}_LIBRARY)
    list(APPEND CLANG_LIBS ${${_lib}_LIBRARY})
    if(${_lib}_LIBRARY)
      target_link_libraries(Clang INTERFACE ${${_lib}_LIBRARY})
    else()
      message(FATAL_ERROR "Could not find ${_lib} library: Try setting CLANG_LIBRARY_DIRS or LLVM_ROOT")
    endif()
  endforeach()
else()
  message(FATAL_ERROR "Could NOT find Clang")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Clang
  REQUIRED_VARS CLANG_INCLUDE_DIRS CLANG_LIBRARY_DIRS CLANG_LIBS
  VERSION_VAR LLVM_VERSION
)
