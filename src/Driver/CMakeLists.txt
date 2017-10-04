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

gtclang_add_library(
  NAME GTClangDriver
  SOURCES CompilerInstance.h
          CompilerInstance.cpp
          Driver.h
          Driver.cpp
          Options.h
          Options.inc
          OptionsParser.h
          OptionsParser.cpp
)

gtclang_add_executable(
  NAME gtclang
  SOURCES gtclang.cpp
  DEPENDS 
    GTClangDriver
    GTClangFrontend
    GTClangSupport
    ${GTCLANG_CLANG_LIBRARIES}
    ${GTCLANG_GSL_LIBRARIES}
    ${GTCLANG_THREADS_LIBRARIES}
)
