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

set(proto_files
  proto/IIR/IIR.proto
)

list(TRANSFORM proto_files REPLACE "[.]proto$" ".pb.cc" OUTPUT_VARIABLE proto_cc)
list(TRANSFORM proto_cc REPLACE "^proto/" "${CMAKE_CURRENT_BINARY_DIR}/")

list(TRANSFORM proto_files REPLACE "[.]proto$" ".pb.h" OUTPUT_VARIABLE proto_h)
list(TRANSFORM proto_h REPLACE "^proto/" "${CMAKE_CURRENT_BINARY_DIR}/")

add_custom_command(
  COMMAND protobuf::protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR} -I ../SIR/proto --proto_path=proto ${proto_files}
  OUTPUT  ${proto_cc} ${proto_h}
  DEPENDS ${proto_files}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating IIR protobuf cpp files"
)

add_library(DawnIIRProto OBJECT ${proto_cc} ${proto_h})
target_compile_features(DawnIIRProto PUBLIC cxx_std_98)
set_target_properties(DawnIIRProto PROPERTIES ${DAWN_TARGET_PROPERTIES})

# Generated files need to be compiled with the protobuf headers and SIR headers
target_include_directories(DawnIIRProto
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

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
  )

target_add_dawn_standard_props(DawnIIR)
target_include_directories(DawnIIR
  PUBLIC $<TARGET_PROPERTY:DawnIIRProto,INCLUDE_DIRECTORIES>
)
target_link_libraries(DawnIIR
  PUBLIC DawnSupport DawnSIR
  PRIVATE DawnIIRProto
)

if(ENABLE_PYTHON)
  list(TRANSFORM proto_files REPLACE "[.]proto$" ".pb.py" OUTPUT_VARIABLE proto_py)
  list(TRANSFORM proto_py PREPEND ${CMAKE_CURRENT_BINARY_DIR}/)

  add_custom_command(
    COMMAND protobuf::protoc --python_out=${PROJECT_BINARY_DIR}/python/dawn -I ../SIR/proto --proto_path=proto ${proto_files}
    OUTPUT  ${proto_py}
    DEPENDS ${proto_files}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating IIR protobuf python files"
  )

#   install(FILES ${iir_proto_python_files} DESTINATION {DAWN_INSTALL_PYTHON_DIR}/IIR)
#   install(FILES ${CMAKE_SOURCE_DIR}/python/IIR/__init__.py DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/IIR)

endif()
