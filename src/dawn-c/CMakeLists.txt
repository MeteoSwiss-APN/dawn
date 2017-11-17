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

dawn_add_library(
  NAME DawnC
  SOURCES Dawn.h
          Compiler.h
          Compiler.cpp
          ErrorHandling.h
          ErrorHandling.cpp
          Options.h
          Options.cpp
          Types.h
          
          util/Allocate.h
          util/OptionsWrapper.h
          util/OptionsWrapper.cpp
  OBJECT
)

# Combine the relevant object libraries into libDawnC.so/libDawnC.a
dawn_combine_libraries(
  NAME DawnC
  OBJECTS DawnCObjects
          DawnSupportObjects
          DawnSIRObjects
          DawnCompilerObjects
          DawnOptimizerObjects
          DawnCodeGenObjects
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  DEPENDS ${DAW_EXTERNAL_LIBRARIES}
)

# Export the targets
install(EXPORT DawnCTargets NAMESPACE Dawn:: DESTINATION ${DAWN_INSTALL_CMAKE_DIR})
