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

set(GTCLANG_UNITTEST_LIBRARIES 
  GTClangUnittest
  GTClangDriver
  GTClangFrontend
  GTClangSupport
  gtest
  ${GTCLANG_EXTERNAL_LIBRARIES}
)

add_subdirectory(Support)
