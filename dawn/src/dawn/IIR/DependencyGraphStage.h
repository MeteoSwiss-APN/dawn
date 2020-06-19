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

#pragma once

#include "dawn/IIR/DependencyGraph.h"

namespace dawn {
class OptimizerContext;

namespace iir {

class Stage;
class StencilInstantiation;

/// @enum DependencyGraphStageEdgeKind
/// @brief Type of edges
/// @ingroup optimizer
enum class DependencyGraphStageEdgeData { Depends };

/// @brief Dependency graph of the stages
/// @ingroup optimizer
class DependencyGraphStage
    : public DependencyGraph<DependencyGraphStage, DependencyGraphStageEdgeData> {

  std::shared_ptr<StencilInstantiation> stencilInstantiation_;

public:
  using Base = DependencyGraph<DependencyGraphStage, DependencyGraphStageEdgeData>;
  using EdgeData = DependencyGraphStageEdgeData;

  DependencyGraphStage(const std::shared_ptr<StencilInstantiation>& stencilInstantiation)
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

} // namespace iir
} // namespace dawn
