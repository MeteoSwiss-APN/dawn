##===------------------------------------------------------------------------------*- CMake -*-===##
##                                 ____ ____  _     
##                                / ___/ ___|| |    
##                               | |  _\___ \| |    
##                               | |_| |___) | |___ 
##                                \____|____/|_____| - Generic Stencil Language
##                  
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

gsl_configure_file(Config.h.cmake)

gsl_add_library(
  NAME GSLSupport
  SOURCES AlignOf.h
          Array.h
          Array.cpp
          ArrayRef.h
          Assert.h
          Assert.cpp
          Casting.h
          Compiler.h
          Config.h.cmake
          EditDistance.h
          Format.h
          FileUtil.h
          FileUtil.cpp
          HashCombine.h
          Json.h
          Logging.h
          Logging.cpp
          MathExtras.h
          Printing.h          
          NonCopyable.h
          SmallVector.h
          SmallVector.cpp
          SourceLocation.h
          SourceLocation.cpp
          STLExtras.h
          StringRef.h
          StringRef.cpp
          StringUtil.h
          StringUtil.cpp
          StringSwitch.h
          SmallString.h
          Type.h
          Type.cpp
          TypeTraits.h
          Twine.h
          Twine.cpp
          UIDGenerator.h
          Unreachable.h
          Unreachable.cpp
)
