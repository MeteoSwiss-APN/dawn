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
add_subdirectory(AST)
add_subdirectory(SIR)
add_subdirectory(Compiler)
add_subdirectory(Optimizer)
add_subdirectory(IIR)
add_subdirectory(CodeGen)
add_subdirectory(Unittest)
add_subdirectory(Serialization)

include(yodaCombineLibraries)
# Combine the relevant object libraries into libDawn.so/libDawn.a
yoda_combine_libraries(
  NAME Dawn
  OBJECTS DawnSupportObjects
          DawnASTObjects
          DawnCompilerObjects
          DawnOptimizerObjects
          DawnCodeGenObjects
          DawnUnittestObjects #TODO: we don't want it here but it is needed for some gtclang tests
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  VERSION ${DAWN_VERSION}
  DEPENDS ${DAWN_EXTERNAL_LIBRARIES} DawnSerializerStatic
)

# Export the targets
install(EXPORT DawnTargets NAMESPACE Dawn:: DESTINATION ${DAWN_INSTALL_CMAKE_DIR})
export(EXPORT DawnTargets NAMESPACE Dawn:: FILE ${PROJECT_BINARY_DIR}/DawnTargets.cmake)
