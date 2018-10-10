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
  NAME DawnIIR
  SOURCES 
          Accesses.cpp  
          Accesses.h
          BlockStatements.cpp
          BlockStatements.h
          Cache.cpp
          Cache.h
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
          StatementAccessesPair.cpp
          StatementAccessesPair.h
          Stencil.cpp
          Stencil.h
#          StencilInstantiation.cpp
#          StencilInstantiation.h
          MetaInformation.cpp
          MetaInformation.h
          StencilFunction/StencilFunctionInstantiation.cpp
          StencilFunction/StencilFunctionInstantiation.h
          StencilFunction/FunctionHandeling.cpp
          StencilFunction/FunctionHandeling.h
  OBJECT
)

target_include_directories(DawnIIRObjects SYSTEM PUBLIC ${Boost_INCLUDE_DIR})
