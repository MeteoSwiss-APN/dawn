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
  NAME DawnSerializer
  SOURCES ASTSerializer.h
          ASTSerializer.cpp
          IIRSerializer.h
          IIRSerializer.cpp
          SIRSerializer.h
          SIRSerializer.cpp
  OBJECT
)

yoda_combine_libraries(
  NAME DawnSerializer
  OBJECTS DawnSerializerObjects
  DEPENDS DawnIIRStatic DawnOptimizerStatic
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  VERSION ${DAWN_VERSION}
  EXPORT_GROUP DawnTargets
)

add_dependencies(DawnSerializerObjects IIR_Proto_Generated)
target_include_directories(DawnSerializerObjects PUBLIC $<TARGET_PROPERTY:DawnIIRStatic,INCLUDE_DIRECTORIES>)


#target_include_directories(DawnSerializerObjects PUBLIC ${sir_proto_include_dirs})
install(EXPORT DawnTargets NAMESPACE Dawn:: DESTINATION ${DAWN_INSTALL_CMAKE_DIR})

