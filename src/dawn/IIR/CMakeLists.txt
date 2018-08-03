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
          DependencyGraph.h
          DependencyGraphAccesses.cpp      
          DependencyGraphAccesses.h
          DependencyGraphStage.cpp
          DependencyGraphStage.h
          DoMethod.cpp
          DoMethod.h
          Field.h
          Field.cpp
          IIRNode.cpp
          IIRNode.h
          MultiStage.cpp 
          MultiStage.h
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
  OBJECT
)

target_include_directories(DawnIIRObjects SYSTEM PUBLIC ${Boost_INCLUDE_DIR})
