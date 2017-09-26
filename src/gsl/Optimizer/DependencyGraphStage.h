//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_OPTIMIZER_DEPENDENCYGRAPHSTAGE_H
#define GSL_OPTIMIZER_DEPENDENCYGRAPHSTAGE_H

#include "gsl/Optimizer/DependencyGraph.h"

namespace gsl {

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

} // namespace gsl

#endif
