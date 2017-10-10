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
# FindClang
# ---------
#
# .. code-block:: cmake
#
#   find_package(Clang [REQUIRED] [QUIET])
#
# Find the Clang_ libraries and includes. Uses the same include and library paths detected 
# by :ref:`FindLLVM`.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# Defines the following variables:
#
# ``CLANG_FOUND``                    
#   True if Clang was found.
# ``CLANG_INCLUDE_DIRS``
#   Where to find Clang includes.
# ``CLANG_LIBS``
#   List of all Clang libraries.
# ``CLANG_LIBCLANG_LIB``
#   C Interface to Clang.
# ``CLANG_LIBRARY_DIRS``
#   Where to find Clang libraries.
# ``CLANG_RESSOURCE_INCLUDE_PATH``
#   Path to the internal Clang headers needed by Frontend tools.
# ``CLANG_<COMPONENT>_LIB``
#   Path to the ``<COMPONENT>`` library.
#
# Hints
# ^^^^^
#
# Set ``LLVM_ROOT`` to a directory that contains a LLVM/Clang installation.
#
# .. _Clang: https://clang.llvm.org/
#

include(CMakeParseArguments)

if(${Clang_FIND_REQUIRED})
  find_package(LLVM ${Clang_FIND_VERSION} REQUIRED)
else()
  find_package(LLVM ${Clang_FIND_VERSION})
endif()

set(CLANG_FOUND FALSE)
set(CLANG_LIBS)

if(LLVM_FOUND AND LLVM_LIBRARY_DIRS)
    
  # Find a clang library
  macro(find_clang_lib libname)
    string(TOUPPER ${libname} prettylibname)
    if(NOT(DEFINED CLANG_${prettylibname}_LIB))
      find_library(CLANG_${prettylibname}_LIB NAMES ${libname} HINTS ${LLVM_LIBRARY_DIRS})
      mark_as_advanced(CLANG_${prettylibname}_LIB)
    endif()    
  endmacro()
  
  # Add a clang library to CLANG_LIBS and resolve dependencies
  macro(add_clang_lib)
    cmake_parse_arguments(_arg "" "NAME" "DEPENDS" ${ARGN})  
    
    find_clang_lib(${_arg_NAME})
    
    string(TOUPPER ${_arg_NAME} prettylibname)    
    if(CLANG_${prettylibname}_LIB)
      list(APPEND CLANG_LIBS ${CLANG_${prettylibname}_LIB})
      
      foreach(dependency ${_arg_DEPENDS})
        string(TOUPPER ${dependency} prettylibname)    
        find_clang_lib(${dependency})
        if(CLANG_${prettylibname}_LIB)
          list(APPEND CLANG_LIBS ${CLANG_${prettylibname}_LIB})          
        endif()
      endforeach()
    endif()
  endmacro()

  find_library(CLANG_LIBCLANG_LIB NAMES clang libclang HINTS ${LLVM_LIBRARY_DIRS})
  mark_as_advanced(CLANG_LIBCLANG_LIB)
  
  # Generate for Clang 3.8.0 
  add_clang_lib(NAME clangRewrite DEPENDS clangBasic clangLex)  
  add_clang_lib(NAME clangDriver DEPENDS clangBasic)
  add_clang_lib(NAME clangSerialization DEPENDS clangAST clangBasic clangLex clangSema)
  add_clang_lib(NAME clangCodeGen DEPENDS clangAST clangBasic clangFrontend clangLex)
  add_clang_lib(NAME clangAST DEPENDS clangBasic clangLex)
  add_clang_lib(NAME clangFrontendTool DEPENDS clangBasic clangCodeGen clangDriver clangFrontend clangRewriteFrontend)
  add_clang_lib(NAME clangTooling DEPENDS clangAST clangASTMatchers clangBasic clangDriver clangFrontend clangLex clangRewrite clangToolingCore)
  add_clang_lib(NAME clangCore DEPENDS clangAST clangBasic clangLex clangRewrite)
  add_clang_lib(NAME clangARCMigrate DEPENDS clangAST clangAnalysis clangBasic clangEdit clangFrontend clangLex clangRewrite clangSema clangSerialization clangStaticAnalyzerCheckers)
  add_clang_lib(NAME clangASTMatchers DEPENDS clangAST clangBasic)
  add_clang_lib(NAME clangDynamic DEPENDS clangAST clangASTMatchers clangBasic)
  add_clang_lib(NAME clangCore DEPENDS clangAST clangAnalysis clangBasic clangLex clangRewrite)
  add_clang_lib(NAME clangCheckers DEPENDS clangAST clangAnalysis clangBasic clangLex clangStaticAnalyzerCore)
  add_clang_lib(NAME clangFrontend DEPENDS clangAST clangAnalysis clangBasic clangFrontend clangLex clangStaticAnalyzerCheckers clangStaticAnalyzerCore)
  add_clang_lib(NAME clangParse DEPENDS clangAST clangBasic clangLex clangSema)
  add_clang_lib(NAME clangSema DEPENDS clangAST clangAnalysis clangBasic clangEdit clangLex)
  add_clang_lib(NAME clangEdit DEPENDS clangAST clangBasic clangLex)
  add_clang_lib(NAME clangIndex DEPENDS clangAST clangBasic clangFormat clangRewrite clangToolingCore)
  add_clang_lib(NAME clangFrontend DEPENDS clangAST clangBasic clangDriver clangEdit clangLex clangParse clangSema clangSerialization)
  add_clang_lib(NAME clangRewrite DEPENDS clangAST clangBasic clangEdit clangFrontend clangLex clangRewrite)
  add_clang_lib(NAME clangAnalysis DEPENDS clangAST clangBasic clangLex)
  add_clang_lib(NAME clangLex DEPENDS clangBasic)
  add_clang_lib(NAME clangBasic)  
  add_clang_lib(NAME clangFormat DEPENDS clangBasic clangLex clangToolingCore)
endif()

if(CLANG_LIBS)
  set(CLANG_FOUND TRUE)
  mark_as_advanced(CLANG_LIBS)
else()
  message(STATUS "Could not find any Clang libraries in ${LLVM_LIBRARY_DIRS}")
endif()

if(CLANG_FOUND)
  set(CLANG_LIBRARY_DIRS ${LLVM_LIBRARY_DIRS})
  set(CLANG_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})

  # Get the ressource path
  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --prefix
    OUTPUT_VARIABLE _llvmInstallPrefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(CLANG_RESSOURCE_INCLUDE_PATH "${_llvmInstallPrefix}/lib/clang/${LLVM_VERSION}/include")
  
  # Check whether llvm-config comes from an install prefix
  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --src-root
    OUTPUT_VARIABLE _llvmSourceRoot
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(FIND "${LLVM_INCLUDE_DIRS}" "${_llvmSourceRoot}" _llvmIsInstalled)
  if(NOT _llvmIsInstalled)
    message(STATUS 
      "Detected that llvm-config comes from a build-tree, adding more include directories for Clang")
    list(APPEND CLANG_INCLUDE_DIRS
         "${LLVM_INSTALL_PREFIX}/tools/clang/include" # build dir
         "${_llvmSourceRoot}/tools/clang/include"     # source dir
    )
  endif()

  message(STATUS "Found Clang (LLVM version: ${LLVM_VERSION})")
  message(STATUS "  Include dirs: ${CLANG_INCLUDE_DIRS}")
  message(STATUS "  Library dirs: ${CLANG_LIBRARY_DIRS}")
else()
  if(Clang_FIND_REQUIRED)
    message(FATAL_ERROR "Could NOT find Clang")
  endif()
endif()