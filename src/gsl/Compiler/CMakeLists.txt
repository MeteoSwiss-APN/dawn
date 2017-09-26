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

gsl_add_library(
  NAME GSLCompiler
  SOURCES DiagnosticsEngine.h
          DiagnosticsEngine.cpp
          DiagnosticsMessage.h
          DiagnosticsQueue.h
          DiagnosticsQueue.cpp
          GSLCompiler.h
          GSLCompiler.cpp
          Options.h
          Options.inc
)
