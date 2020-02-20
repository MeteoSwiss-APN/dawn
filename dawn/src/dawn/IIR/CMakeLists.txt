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

# Defines dawn_iir_proto_files
include(proto/DawnIIR.cmake)

include(DawnProtobufGenerate)
dawn_protobuf_generate(
  OUT_FILES iir_proto_cpp_files
  PROTOS ${dawn_iir_proto_files}
  INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../SIR/proto/
  WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
  PACKG IIR
  LANGUAGE cpp
)

add_library(DawnIIRProto OBJECT ${iir_proto_cpp_files})
target_compile_features(DawnIIRProto PUBLIC cxx_std_11)
set_target_properties(DawnIIRProto PROPERTIES ${DAWN_TARGET_PROPERTIES})

# Generated files need to be compiled with the protobuf headers and SIR headers
target_include_directories(DawnIIRProto
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# ... and linked to protobuf
target_link_libraries(DawnIIRProto PUBLIC DawnSIRProto protobuf::libprotobuf)

add_library(DawnIIR
  AccessComputation.h
  AccessComputation.cpp
  AST.h
  ASTConverter.cpp
  ASTConverter.h
  ASTExpr.cpp
  ASTExpr.h
  ASTFwd.h
  ASTStmt.h
  ASTStmt.cpp
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
  IIRNode.h
  IIRNodeIterator.h
  LocalVariable.cpp
  LocalVariable.h
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
  Stencil.cpp
  Stencil.h
  StencilFunctionInstantiation.cpp
  StencilFunctionInstantiation.h
  StencilInstantiation.cpp
  StencilInstantiation.h
  StencilMetaInformation.cpp
  StencilMetaInformation.h
  $<TARGET_OBJECTS:DawnIIRProto>
  )

target_add_dawn_standard_props(DawnIIR)
target_link_libraries(DawnIIR PUBLIC DawnSupport DawnSIR DawnIIRProto)

# TODO add this if CLAW needs it (-> Matthias)
# if(DAWN_JAVA)
#   # Genreate python proto files
#   dawn_protobuf_generate(
#     OUT_FILES iir_proto_java_files
#     OUT_DIRS iir_proto_java_dirs
#     PROTOS IIR/IIR.proto
#     INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../SIR/proto/
#     WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
#     PACKG IIR
#     LANGUAGE java
#   )
#   add_custom_target(DawnIIRJavaProto ALL
#     COMMAND ${CMAKE_COMMAND} -E copy ${iir_proto_java_dirs} ${CMAKE_BINARY_DIR}/java/dawn
#     DEPENDS ${iir_proto_java_files}
#   )
#   install(DIRECTORY ${iir_proto_java_dirs} DESTINATION ${DAWN_INSTALL_JAVA_DIR}/dawn)
# endif(DAWN_JAVA)
