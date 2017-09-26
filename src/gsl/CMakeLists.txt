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

add_subdirectory(Support)
add_subdirectory(SIR)
add_subdirectory(Compiler)
add_subdirectory(Optimizer)
add_subdirectory(CodeGen)
add_subdirectory(Unittest)

# Combine the relevant object libraries into libGSL.so/libGSL.a
gsl_combine_libraries(
  NAME GSL
  OBJECTS GSLSupportObjects
          GSLSIRObjects
          GSLCompilerObjects
          GSLOptimizerObjects
          GSLCodeGenObjects
  DEPENDS ${METIS_LIBRARY}
)
