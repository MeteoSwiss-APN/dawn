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
#include "dawn/Optimizer/BoundaryExtent.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/DependencyGraphStage.h"
#include "dawn/Optimizer/MultiStage.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Optimizer/Stencil.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include <algorithm>
#include <set>
#include <vector>

namespace dawn {

/// @brief Check if we can merge the stage into the multi-stage, possibly changing the loop order.
/// @returns the the enew dependency graphs of the multi-stage (or NULL) and the new loop order
template <typename ReturnType = std::pair<std::shared_ptr<DependencyGraphAccesses>, LoopOrderKind>>
ReturnType isMergable(const Stage& stage, LoopOrderKind stageLoopOrder,
                      const MultiStage& multiStage) {
  LoopOrderKind multiStageLoopOrder = multiStage.getLoopOrder();
  auto multiStageDependencyGraph =
      multiStage.getDependencyGraphOfInterval(stage.getEnclosingExtendedInterval());

  // Merge stage into dependency graph
  const DoMethod& doMethod = stage.getSingleDoMethod();
  multiStageDependencyGraph->merge(doMethod.getDependencyGraph().get());

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
  std::vector<LoopOrderKind> possibleLoopOrders;

  if(multiStageLoopOrder == LoopOrderKind::LK_Parallel &&
     stageLoopOrder == LoopOrderKind::LK_Parallel)
    possibleLoopOrders = {LoopOrderKind::LK_Parallel, LoopOrderKind::LK_Forward,
                          LoopOrderKind::LK_Backward};
  else if(stageLoopOrder == LoopOrderKind::LK_Parallel)
    possibleLoopOrders.push_back(multiStageLoopOrder);
  else
    possibleLoopOrders.push_back(stageLoopOrder);

  if(multiStageDependencyGraph->empty())
    return ReturnType(multiStageDependencyGraph, possibleLoopOrders.front());

  // If the resulting graph isn't a DAG anymore that isn't gonna work
  if(!multiStageDependencyGraph->isDAG())
    return ReturnType(nullptr, multiStageLoopOrder);

  // Check all possible loop orders if there aren't any vertical conflicts
  for(auto loopOrder : possibleLoopOrders) {
    auto conflict = hasVerticalReadBeforeWriteConflict(multiStageDependencyGraph.get(), loopOrder);
    if(!conflict.CounterLoopOrderConflict)
      return ReturnType(multiStageDependencyGraph, loopOrder);
  }

  return ReturnType(nullptr, multiStageLoopOrder);
}

std::shared_ptr<Stencil> ReoderStrategyGreedy::reorder(const std::shared_ptr<Stencil>& stencilPtr) {
  Stencil& stencil = *stencilPtr;

  DependencyGraphStage& stageDAG = *stencil.getStageDependencyGraph();
  StencilInstantiation* instantiation = stencil.getStencilInstantiation();

  std::shared_ptr<Stencil> newStencil =
      std::make_shared<Stencil>(instantiation, stencil.getSIRStencil(), stencilPtr->getStencilID(),
                                stencil.getStageDependencyGraph());
  int newNumStages = 0;
  int newNumMultiStages = 0;

  const int maxBoundaryExtent = instantiation->getOptimizerContext()->getOptions().MaxHaloPoints;

  auto pushBackNewMultiStage = [&](LoopOrderKind loopOrder) -> void {
    newStencil->getMultiStages().push_back(std::make_shared<MultiStage>(instantiation, loopOrder));
    newNumMultiStages++;
  };

  for(const auto& multiStagePtr : stencil.getMultiStages()) {

    // First time we encounter this multi-stage, create an empty multi-stage
    pushBackNewMultiStage(LoopOrderKind::LK_Parallel);

    for(const auto& stagePtr : multiStagePtr->getStages()) {
      const Stage& stage = *stagePtr;
      int stageIdx = newNumStages - 1;

      // Compute the best possible position to where we can move this stage without violating
      // any dependencies
      for(; stageIdx >= 0; --stageIdx) {
        if(stageDAG.depends(stage.getStageID(), newStencil->getStage(stageIdx)->getStageID()))
          break;
      }

      Stencil::StagePosition pos = newStencil->getPositionFromStageIndex(stageIdx);
      LoopOrderKind stageLoopOrder = multiStagePtr->getLoopOrder();

      // Find the first available multi-stage
      bool lastChance = false;
      while(true) {
        const auto& MS = newStencil->getMultiStageFromMultiStageIndex(pos.MultiStageIndex);

        // 1) Are the loop orders compatible?
        if(loopOrdersAreCompatible(stageLoopOrder, MS->getLoopOrder())) {

          // 2) Can we merge the stage wihtout violating vertical dependencies?
          auto dependencyGraphLoopOrderPair = isMergable(stage, stageLoopOrder, *MS);
          auto multiStageDependencyGraph = dependencyGraphLoopOrderPair.first;

          if(multiStageDependencyGraph) {

            // 3) Do we not exceed the maximum allowed boundary extents?
            if(!exceedsMaxBoundaryPoints(multiStageDependencyGraph.get(), maxBoundaryExtent)) {

              // Yes, Yes and Yes ... stop and insert the stage!
              MS->setLoopOrder(dependencyGraphLoopOrderPair.second);
              break;
            } else if(lastChance) {
              // Our stage exceeds the maximum allowed boundary extents... nothing we can do
              DiagnosticsBuilder diag(DiagnosticsKind::Error, SourceLocation());
              diag << "stencil '" << instantiation->getName()
                   << "' exceeds maximum number of allowed halo lines (" << maxBoundaryExtent
                   << ")";
              instantiation->getOptimizerContext()->getDiagnostics().report(diag);
              return nullptr;
            }
          }
          DAWN_ASSERT_MSG(!lastChance,
                          "merging stage in empty multi-stage failed (this probably means the "
                          "stage graph contains cycles - i.e is not a DAG!)");
        }

        // Advance to the next multi-stage
        pos.MultiStageIndex++;
        pos.StageOffset = -1;

        // The last available multi-stage wasn't legal, we push-back a new multistage with parallel
        // loop order (this will guarantee a success the next check if our stage does not exceed the
        // maximum boundary lines in which case we abort)
        if(pos.MultiStageIndex == newNumMultiStages) {
          pushBackNewMultiStage(LoopOrderKind::LK_Parallel);
          lastChance = true;
        }
      }

      newNumStages++;
      newStencil->insertStage(pos, stagePtr);
    }
  }

  // Remove empty multi-stages
  for(auto it = newStencil->getMultiStages().begin(); it != newStencil->getMultiStages().end();) {
    if((*it)->getStages().empty())
      it = newStencil->getMultiStages().erase(it);
    else
      it++;
  }

  return newStencil;
}

} // namespace dawn
