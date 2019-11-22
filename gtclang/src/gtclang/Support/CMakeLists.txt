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

target_link_libraries(GTClangSupport PUBLIC Dawn::Dawn Clang)

# include(yodaConfigureFile)
# include(yodaAddLibrary)

# yoda_configure_file(Config.h.cmake)

# yoda_add_library(
#   NAME GTClangSupport
#   SOURCES FileUtil.cpp
#           FileUtil.h
#           Logger.cpp
#           Logger.h
#           StringUtil.cpp
#           StringUtil.h
#           ASTUtils.cpp
#           ASTUtils.h
#   OBJECT
# )
