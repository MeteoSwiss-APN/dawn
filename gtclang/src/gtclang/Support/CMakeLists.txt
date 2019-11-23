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

configure_file(Config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/Config.h)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/Config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dawn/Support
)

add_library(GTClangSupport
  FileUtil.cpp
  FileUtil.h
  Logger.cpp
  Logger.h
  StringUtil.cpp
  StringUtil.h
  ASTUtils.cpp
  ASTUtils.h
)

target_include_directories(GTClangSupport
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_add_gtclang_standard_props(GTClangSupport)
target_link_libraries(GTClangSupport PUBLIC Dawn::DawnSupport clangAST)
