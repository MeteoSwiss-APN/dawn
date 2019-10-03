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

include(yodaConfigureFile)
include(yodaAddLibrary)

yoda_configure_file(Config.h.cmake)

yoda_add_library(
  NAME GTClangUnittest
  SOURCES FileManager.cpp
          FileManager.h
          FlagManager.cpp
          FlagManager.h
          GTClang.cpp
          GTClang.h
          UnittestEnvironment.cpp
          UnittestEnvironment.h
          UnittestStmtSimplifier.h
          UnittestStmtSimplifier.cpp          
          ParsingComparison.h
          ParsingComparison.cpp
          ParsedString.h
          ParsedString.cpp
  ARCHIVE
)

