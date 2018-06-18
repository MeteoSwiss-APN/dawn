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

include(mchbuildCombineLibraries)
# Combine the relevant object libraries into libDawn.so/libDawn.a
mchbuild_combine_libraries(
  NAME Dawn
  OBJECTS DawnSupportObjects
          DawnSIRObjects
          DawnCompilerObjects
          DawnOptimizerObjects
          DawnCodeGenObjects
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  VERSION ${DAWN_VERSION}
  DEPENDS ${DAWN_EXTERNAL_LIBRARIES}
)

target_include_directories(DawnShared SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})

# Export the targets
install(EXPORT DawnTargets NAMESPACE Dawn:: DESTINATION ${DAWN_INSTALL_CMAKE_DIR})
