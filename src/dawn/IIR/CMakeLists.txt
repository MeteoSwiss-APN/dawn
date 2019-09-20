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


include(DawnProtobufGenerate)

# Genreate C++ proto files
dawn_protobuf_generate(
  OUT_FILES iir_proto_cpp_files
  OUT_INCLUDE_DIRS iir_proto_include_dirs
  PROTOS IIR/IIR.proto
  INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../SIR/proto/
  WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
  PACKG IIR
  LANGUAGE cpp
)

yoda_add_library(
  NAME DawnIIR
  SOURCES AST.h
          ASTExpr.h
          ASTFwd.h
          ASTStmt.h
          ASTStringifier.h
          ASTUtil.h
          ASTUtil.cpp
          ASTVisitor.h
          Accesses.cpp  
          Accesses.h
          AccessUtils.cpp
          AccessUtils.h
          AccessToNameMapper.cpp
          AccessToNameMapper.h
          BlockStatements.cpp
          BlockStatements.h
          Cache.cpp
          Cache.h
          ControlFlowDescriptor.cpp
          ControlFlowDescriptor.h
          DependencyGraph.h
          DependencyGraphAccesses.cpp      
          DependencyGraphAccesses.h
          DependencyGraphStage.cpp
          DependencyGraphStage.h
          DoMethod.cpp
          DoMethod.h
          Extents.cpp
          Extents.h
          Field.h
          Field.cpp
          FieldAccessExtents.cpp
          FieldAccessExtents.h
          FieldAccessMetadata.cpp
          FieldAccessMetadata.h
          InstantiationHelper.cpp
          InstantiationHelper.h
          Interval.cpp
          Interval.h
          IntervalAlgorithms.cpp
          IntervalAlgorithms.h
          IIR.cpp
          IIR.h
          IIRBuilder.cpp
          IIRBuilder.h
          proto/IIR/IIR.proto
          IIRNode.h
          IIRNodeIterator.h
          LoopOrder.cpp
          LoopOrder.h
          MultiInterval.cpp
          MultiInterval.h
          MultiStage.cpp 
          MultiStage.h
          NodeUpdateType.cpp
          NodeUpdateType.h
          Stage.cpp
          Stage.h
          StatementAccessesPair.cpp
          StatementAccessesPair.h
          Stencil.cpp
          Stencil.h
          StencilFunctionInstantiation.cpp
          StencilFunctionInstantiation.h
          StencilInstantiation.cpp
          StencilInstantiation.h
          StencilMetaInformation.cpp
          StencilMetaInformation.h
          ${iir_proto_cpp_files}
  OBJECT
)
target_include_directories(DawnIIRObjects PUBLIC ${iir_proto_include_dirs}/../)

target_include_directories(DawnIIRObjects PUBLIC $<TARGET_PROPERTY:DawnSIRStatic,INCLUDE_DIRECTORIES>)
target_include_directories(DawnIIRObjects SYSTEM PUBLIC ${Boost_INCLUDE_DIR} ${iir_proto_include_dirs})

add_custom_target(IIR_Proto_Generated ALL DEPENDS ${iir_proto_cpp_files})
add_dependencies(IIR_Proto_Generated SIR_Proto_Generated)

yoda_combine_libraries(
  NAME DawnIIR
  OBJECTS DawnIIRObjects
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  DEPENDS DawnSIRStatic
  VERSION ${DAWN_VERSION}
  EXPORT_GROUP DawnTargets
)

if(DAWN_PYTHON) 
  # Genreate python proto files
  dawn_protobuf_generate(
    OUT_FILES iir_proto_python_files
    PROTOS IIR/IIR.proto
    INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../SIR/proto/
    WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PACKG IIR
    LANGUAGE python
  )

  add_custom_target(DawnIIRPythonProto ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${iir_proto_python_files} ${CMAKE_BINARY_DIR}/python/dawn
    DEPENDS ${iir_proto_python_files}
  )
  install(FILES ${iir_proto_python_files} DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/IIR)
  install(FILES ${CMAKE_SOURCE_DIR}/python/IIR/__init__.py DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/IIR)
endif(DAWN_PYTHON)

if(DAWN_JAVA)
  # Genreate python proto files
  dawn_protobuf_generate(
    OUT_FILES iir_proto_java_files
    OUT_DIRS iir_proto_java_dirs
    PROTOS IIR/IIR.proto
    INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../SIR/proto/
    WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PACKG IIR
    LANGUAGE java
  )

  add_custom_target(DawnIIRJavaProto ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${iir_proto_java_dirs} ${CMAKE_BINARY_DIR}/java/dawn
    DEPENDS ${iir_proto_java_files}
  )

  install(DIRECTORY ${iir_proto_java_dirs} DESTINATION ${DAWN_INSTALL_JAVA_DIR}/dawn)
endif(DAWN_JAVA)

