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
  PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/IIR.proto
  LANGUAGE cpp
)

yoda_add_library(
  NAME DawnIIR
  SOURCES 
          Accesses.cpp  
          Accesses.h
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
          IIR.proto
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

