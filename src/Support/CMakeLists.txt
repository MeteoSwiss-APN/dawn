##===------------------------------------------------------------------------------*- CMake -*-===##
##                         _     _ _              _            _                   
##                        (_)   | | |            | |          | |                  
##               __ _ _ __ _  __| | |_ ___   ___ | |___    ___| | __ _ _ __   __ _ 
##              / _` | '__| |/ _` | __/ _ \ / _ \| / __|  / __| |/ _` | '_ \ / _` |
##             | (_| | |  | | (_| | || (_) | (_) | \__ \ | (__| | (_| | | | | (_| |
##              \__, |_|  |_|\__,_|\__\___/ \___/|_|___/  \___|_|\__,_|_| |_|\__, |
##               __/ |                                                        __/ |
##              |___/                                                        |___/ 
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

gtclang_configure_file(Config.h.cmake)

gtclang_add_library(
  NAME GTClangSupport
  SOURCES StringUtil.h
          StringUtil.cpp
          Logger.h
          Logger.cpp
          FileUtil.h
          FileUtil.cpp
)
