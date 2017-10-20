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

dawn_configure_file(Config.h.cmake)

dawn_add_library(
  NAME GTClangUnittest
  SOURCES FileManager.cpp
          FileManager.h
          FlagManager.cpp
          FlagManager.h
          GTClang.cpp
          GTClang.h
          UnittestEnvironment.cpp
          UnittestEnvironment.h
          UnittestStmtSimplyfier.h
          UnittestStmtSimplyfier.cpp          
          SirGenerator.h
          SirGenerator.cpp
          ParsedString.h
          ParsedString.cpp
  ARCHIVE
)

