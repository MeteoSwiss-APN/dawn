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

add_subdirectory(Support)
add_subdirectory(SIR)
add_subdirectory(Compiler)
add_subdirectory(Optimizer)
add_subdirectory(CodeGen)
add_subdirectory(Unittest)

# Combine the relevant object libraries into libdawn.so/libdawn.a
dawn_combine_libraries(
  NAME Dawn
  OBJECTS DawnSupportObjects
          DawnSIRObjects
          DawnCompilerObjects
          DawnOptimizerObjects
          DawnCodeGenObjects
)
