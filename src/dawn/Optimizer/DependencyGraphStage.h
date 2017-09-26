//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _                      
//                         | |                     
//                       __| | __ ___      ___ ___  
//                      / _` |/ _` \ \ /\ / / '_  | 
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT). 
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef DAWN_OPTIMIZER_DEPENDENCYGRAPHSTAGE_H
#define DAWN_OPTIMIZER_DEPENDENCYGRAPHSTAGE_H

#include "dawn/Optimizer/DependencyGraph.h"

namespace dawn {

class Stage;
class StencilInstantiation;
class OptimizerContext;

/// @enum DependencyGraphStageEdgeKind
/// @brief Type of edges
/// @ingroup optimizer
enum class DependencyGraphStageEdgeData { EK_Depends };

/// @brief Dependency graph of the stages
/// @ingroup optimizer
class DependencyGraphStage
    : public DependencyGraph<DependencyGraphStage, DependencyGraphStageEdgeData> {

  StencilInstantiation* stencilInstantiation_;

public:
  using Base = DependencyGraph<DependencyGraphStage, DependencyGraphStageEdgeData>;
  using EdgeData = DependencyGraphStageEdgeData;

  DependencyGraphStage(StencilInstantiation* stencilInstantiation)
      : Base(), stencilInstantiation_(stencilInstantiation) {}

  void insertEdge(int StageIDFrom, int StageIDTo);

  /// Check if stage `From` depends on stage `To`
  bool depends(int StageIDFrom, int StageIDTo) const;

  /// @brief EdgeData to string
  const char* edgeDataToString(const EdgeData& data) const;

  /// @brief EdgeData to dot
  std::string edgeDataToDot(const EdgeData& data) const;

  /// @brief Get the name of the vertex given by ID
  std::string getVertexNameByVertexID(std::size_t VertexID) const;

  /// @brief Get the shape of the dot grahps
  const char* getDotShape() const;
};

} // namespace dawn

#endif
