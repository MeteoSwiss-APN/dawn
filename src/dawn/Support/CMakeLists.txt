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

mchbuild_configure_file(Config.h.cmake)

mchbuild_add_library(
  NAME DawnSupport
  SOURCES AlignOf.h
          Array.cpp
          Array.h
          ArrayRef.h
          Assert.cpp
          Assert.h
          Casting.h
          Compiler.h
          Config.h.cmake
          EditDistance.h
          FileUtil.cpp
          FileUtil.h
          Format.h
          HashCombine.h
          Json.h
          Logging.cpp
          Logging.h
          MathExtras.h
          NonCopyable.h
          Printing.h          
          SmallString.h
          SmallVector.cpp
          SmallVector.h
          SourceLocation.cpp
          SourceLocation.h
          STLExtras.h
          StringRef.cpp
          StringRef.h
          StringSwitch.h
          StringUtil.cpp
          StringUtil.h
          Twine.cpp
          Twine.h
          Type.cpp
          Type.h
          TypeTraits.h
          UIDGenerator.h
          Unreachable.cpp
          Unreachable.h
          ../Dawn.h
  OBJECT
)
