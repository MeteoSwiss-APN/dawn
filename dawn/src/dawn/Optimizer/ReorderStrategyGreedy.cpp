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

#include "dawn/Optimizer/ReorderStrategyGreedy.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Support/Exception.h"
#include <algorithm>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

namespace dawn {

/// @brief Check if we can merge the stage into the multi-stage, possibly changing the loop order.
/// @returns the the enew dependency graphs of the multi-stage (or NULL) and the new loop order
template <typename ReturnType =
              std::pair<std::optional<iir::DependencyGraphAccesses>, iir::LoopOrderKind>>
ReturnType isMergable(const iir::Stage& stage, iir::LoopOrderKind stageLoopOrder,
                      const iir::MultiStage& multiStage) {
  iir::LoopOrderKind multiStageLoopOrder = multiStage.getLoopOrder();
  auto multiStageDependencyGraph =
      multiStage.getDependencyGraphOfInterval(stage.getEnclosingExtendedInterval());

  // Merge stage into dependency graph
  const iir::DoMethod& doMethod = stage.getSingleDoMethod();
  multiStageDependencyGraph.merge(*doMethod.getDependencyGraph());

  // Try all possible loop orders while *favoring* a parallel loop order. Note that a parallel loop
  // order can be changed to forward or backward.
  //
  //                 MULTI-STAGE
  //
  //             |  P  |  F  |  B  |           P = Parallel
  //        -----+-----+-----+-----+           F = Foward
  //    S     P  | PFB    F     B  |           B = Backward
  //    T   -----+                 +           X = Incompatible
  //    A     F  |  F     F     X  |
  //    G   -----+                 +
  //    E     B  |  B     X     B  |
  //        -----+-----------------+
  //
  std::vector<iir::LoopOrderKind> possibleLoopOrders;

  if(multiStageLoopOrder == iir::LoopOrderKind::Parallel &&
     stageLoopOrder == iir::LoopOrderKind::Parallel)
    possibleLoopOrders = {iir::LoopOrderKind::Parallel, iir::LoopOrderKind::Forward,
                          iir::LoopOrderKind::Backward};
  else if(stageLoopOrder == iir::LoopOrderKind::Parallel)
    possibleLoopOrders.push_back(multiStageLoopOrder);
  else
    possibleLoopOrders.push_back(stageLoopOrder);

  if(multiStageDependencyGraph.empty())
    return ReturnType(multiStageDependencyGraph, possibleLoopOrders.front());

  // If the resulting graph isn't a DAG anymore that isn't gonna work
  if(!multiStageDependencyGraph.isDAG())
    return ReturnType(std::nullopt, multiStageLoopOrder);

  // Check all possible loop orders if there aren't any vertical conflicts
  for(auto loopOrder : possibleLoopOrders) {
    auto conflict = hasVerticalReadBeforeWriteConflict(multiStageDependencyGraph, loopOrder);
    if(!conflict.CounterLoopOrderConflict)
      return ReturnType(multiStageDependencyGraph, loopOrder);
  }

  return ReturnType(std::nullopt, multiStageLoopOrder);
}

std::unique_ptr<iir::Stencil>
ReoderStrategyGreedy::reorder(iir::StencilInstantiation* instantiation,
                              const std::unique_ptr<iir::Stencil>& stencilPtr,
                              const Options& options) {
  iir::Stencil& stencil = *stencilPtr;

  auto const& stageDAG = *stencil.getStageDependencyGraph();

  auto& metadata = instantiation->getMetaData();
  std::unique_ptr<iir::Stencil> newStencil = std::make_unique<iir::Stencil>(
      metadata, stencil.getStencilAttributes(), stencilPtr->getStencilID());

  newStencil->setStageDependencyGraph(iir::DependencyGraphStage(stageDAG));
  int newNumStages = 0;
  int newNumMultiStages = 0;

  const int maxBoundaryExtent = options.MaxHaloPoints;

  auto pushBackNewMultiStage = [&](iir::LoopOrderKind loopOrder) -> void {
    newStencil->insertChild(std::make_unique<iir::MultiStage>(metadata, loopOrder));
    newNumMultiStages++;
  };

  for(const auto& multiStagePtr : stencil.getChildren()) {

    // First time we encounter this multi-stage, create an empty multi-stage
    pushBackNewMultiStage(iir::LoopOrderKind::Parallel);

    for(auto& stagePtr : multiStagePtr->getChildren()) {
      const iir::Stage& stage = *stagePtr;
      int stageIdx = newNumStages - 1;

      // Compute the best possible position to where we can move this stage without violating
      // any dependencies
      for(; stageIdx >= 0; --stageIdx) {
        if(stageDAG.depends(stage.getStageID(), newStencil->getStage(stageIdx)->getStageID()))
          break;
      }

      iir::Stencil::StagePosition pos = newStencil->getPositionFromStageIndex(stageIdx);
      iir::LoopOrderKind stageLoopOrder = multiStagePtr->getLoopOrder();

      // Find the first available multi-stage
      bool lastChance = false;
      while(true) {
        const auto& MS = newStencil->getMultiStageFromMultiStageIndex(pos.MultiStageIndex);

        // 1) Are the loop orders compatible?
        if(loopOrdersAreCompatible(stageLoopOrder, MS->getLoopOrder())) {

          // 2) Can we merge the stage without violating vertical dependencies?
          auto dependencyGraphLoopOrderPair = isMergable(stage, stageLoopOrder, *MS);
          auto multiStageDependencyGraph = dependencyGraphLoopOrderPair.first;

          if(multiStageDependencyGraph) {

            // 3) Do we not exceed the maximum allowed boundary extents?
            if(!multiStageDependencyGraph->exceedsMaxBoundaryPoints(maxBoundaryExtent)) {

              // Yes, Yes and Yes ... stop and insert the stage!
              MS->setLoopOrder(dependencyGraphLoopOrderPair.second);
              break;
            } else if(lastChance) {
              // Our stage exceeds the maximum allowed boundary extents... nothing we can do
              std::stringstream ss;
              ss << "Stencil '" << instantiation->getName()
                 << "' exceeds maximum number of allowed halo lines (" << maxBoundaryExtent << ")";
              throw SemanticError(ss.str());
            }
          }
          if(lastChance) {
            throw SemanticError(
                "merging stage in empty multi-stage failed (this probably means the "
                "stage graph contains cycles - i.e is not a DAG!)");
          }
        }

        // Advance to the next multi-stage
        pos.MultiStageIndex++;
        pos.StageOffset = -1;

        // The last available multi-stage wasn't legal, we push-back a new multistage with parallel
        // loop order (this will guarantee a success the next check if our stage does not exceed the
        // maximum boundary lines in which case we abort)
        if(pos.MultiStageIndex == newNumMultiStages) {
          pushBackNewMultiStage(iir::LoopOrderKind::Parallel);
          lastChance = true;
        }
      }

      newNumStages++;
      newStencil->insertStage(pos, std::move(stagePtr));
    }
  }

  // Remove empty multi-stages
  for(auto it = newStencil->childrenBegin(); it != newStencil->childrenEnd();) {
    if((*it)->childrenEmpty())
      it = newStencil->childrenErase(it);
    else
      it++;
  }

  return newStencil;
}

} // namespace dawn
