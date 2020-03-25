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

#include "dawn/Optimizer/PassMultiStageMerger.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Support/Iterator.h"

#include <map>

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

/// @brief Check if we can merge the stage into the multi-stage, possibly changing the loop order.
/// @returns the the enew dependency graphs of the multi-stage (or NULL) and the new loop order
template <typename ReturnType =
              std::pair<std::optional<iir::DependencyGraphAccesses>, iir::LoopOrderKind>>
ReturnType isMergable(const iir::MultiStage& thisMS, const iir::MultiStage& otherMS) {
  iir::LoopOrderKind thisLoopOrder = thisMS.getLoopOrder();
  for(const auto& thisStage : thisMS.getChildren()) {
    auto dependencyGraphLoopOrderPair = isMergable(*thisStage, thisLoopOrder, otherMS);
    if(dependencyGraphLoopOrderPair.first) {
      return dependencyGraphLoopOrderPair;
    }
  }
  return ReturnType(std::nullopt, thisLoopOrder);
}

PassMultiStageMerger::PassMultiStageMerger(OptimizerContext& context)
    : Pass(context, "PassMultiStageMerger") {
  dependencies_.push_back("PassSetStageGraph");
}

bool PassMultiStageMerger::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  // Do we need to run this Pass?
  bool doMultiStageMerge = false;
  for(const auto& stencil : instantiation->getStencils()) {
    doMultiStageMerge |= stencil->getChildren().size() > 1;
    if(doMultiStageMerge)
      break;
  }

  if(!doMultiStageMerge)
    return true;

  const int maxBoundaryExtent = context_.getOptions().MaxHaloPoints;

  for(const auto& stencil : instantiation->getStencils()) {
    unsigned nMultiStages = stencil->getChildren().size();
    if(nMultiStages < 2)
      continue;

    auto& metadata = instantiation->getMetaData();
    std::unique_ptr<iir::Stencil> newStencil = std::make_unique<iir::Stencil>(
        metadata, stencil->getStencilAttributes(), stencil->getStencilID());

    // Note that the underlying assumption is that stages in the same multi-stage are guaranteed to
    // have no counter loop-oorder vertical dependencies. We can thus treat each multi-stage in
    // isolation!
    for(auto [thisIdx, thisMS] : enumerate(stencil->getChildren())) {
      int mergeIdx = -1;
      for(auto [otherIdx, otherMS] : enumerate(newStencil->getChildren())) {
        // 1) Are the loop orders compatible?
        if(loopOrdersAreCompatible(thisMS->getLoopOrder(), otherMS->getLoopOrder())) {
          // 2) Can we merge the stage without violating vertical dependencies?
          auto dependencyGraphLoopOrderPair = isMergable(*thisMS, *otherMS);
          auto multiStageDependencyGraph = dependencyGraphLoopOrderPair.first;
          if(multiStageDependencyGraph &&
             !multiStageDependencyGraph->exceedsMaxBoundaryPoints(maxBoundaryExtent)) {
            otherMS->setLoopOrder(dependencyGraphLoopOrderPair.second);
            mergeIdx = otherIdx;
          }
        }
      }

      if(mergeIdx < 0) {
        mergeIdx = newStencil->getChildren().size();
        newStencil->insertChild(
            std::make_unique<iir::MultiStage>(metadata, thisMS->getLoopOrder()));
      }

      const auto& mergeMS = newStencil->getMultiStageFromMultiStageIndex(mergeIdx);
      int stageIdx = mergeMS->getChildren().size() - 1;
      for(auto& thisStage : thisMS->getChildren()) {
        iir::Stencil::StagePosition pos(mergeIdx, stageIdx);
        newStencil->insertStage(pos, std::move(thisStage));
        stageIdx += 1;
      }
    }

    instantiation->getIIR()->replace(stencil, newStencil, instantiation->getIIR());
  }

  return true;
}

} // namespace dawn
