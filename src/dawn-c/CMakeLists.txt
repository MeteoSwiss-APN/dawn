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

yoda_add_library(
  NAME DawnC
  SOURCES Dawn.h
          Compiler.cpp
          Compiler.h
          ErrorHandling.cpp
          ErrorHandling.h
          Options.cpp
          Options.h
          TranslationUnit.cpp
          TranslationUnit.h
          Types.h
          
          util/Allocate.h
          util/CompilerWrapper.h          
          util/OptionsWrapper.cpp
          util/OptionsWrapper.h
          util/TranslationUnitWrapper.h
  OBJECT
)

# Combine the relevant object libraries into libDawnC.so/libDawnC.a
yoda_combine_libraries(
  NAME DawnC
  OBJECTS DawnCObjects
          DawnSupportObjects
          DawnSIRObjects
          DawnCompilerObjects
          DawnIIRObjects
          DawnOptimizerObjects
          DawnCodeGenObjects
          DawnSerializerObjects
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  VERSION ${DAWN_VERSION}
  DEPENDS ${DAWN_EXTERNAL_LIBRARIES}
)

# Export the targets
install(EXPORT DawnCTargets NAMESPACE Dawn:: DESTINATION ${DAWN_INSTALL_CMAKE_DIR})
