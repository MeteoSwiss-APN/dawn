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
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Support/FileUtil.h"

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
  for(const auto& otherStage : otherMS.getChildren()) {
    auto dependencyGraphLoopOrderPair = isMergable(*otherStage, thisLoopOrder, thisMS);
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
    if(stencil->getChildren().size() < 2)
      continue;

    auto stencilAxis = stencil->getAxis(false);
    const auto& stageDAG = stencil->getStageDependencyGraph().value();

    // Note that the underlying assumption is that stages in the same multi-stage are guaranteed to
    // have no counter loop-oorder vertical dependencies. We can thus treat each multi-stage in
    // isolation!
    for(const auto& thisMS : stencil->getChildren()) {
      for(const auto& otherMS : stencil->getChildren()) {
        // 1) Are the loop orders compatible?
        if(thisMS->getID() != otherMS->getID() &&
           loopOrdersAreCompatible(thisMS->getLoopOrder(), otherMS->getLoopOrder())) {

          // 2) Can we merge the stage without violating vertical dependencies?
          auto dependencyGraphLoopOrderPair =
              isMergable(*thisMS, *otherMS); // otherStage, thisLoopOrder, *thisMS);
          auto multiStageDependencyGraph = dependencyGraphLoopOrderPair.first;
          if(multiStageDependencyGraph &&
             !multiStageDependencyGraph->exceedsMaxBoundaryPoints(maxBoundaryExtent)) {
            int stop = 1;
          }

          //          bool dependsOn = multiStageDependsOn(thisMS, otherMS, stageDAG);
          //          bool doMerge = !dependsOn;
          //
          //          if(dependsOn) {
          //            // Do loop order analysis...
          //          }
        }
      }
      // Iterate stages backwards (bottom -> top)
      //      for(auto curStageIt = multiStage.childrenRBegin(); curStageIt !=
      //      multiStage.childrenREnd();) {
      //
      //        iir::Stage& curStage = **curStageIt;
      //
      //        bool updateFields = false;
      //
      //        // If our Do-Methods already spans the entire axis, we don't want to destroy that
      //        property bool MergeDoMethodsOfStage = curStage.getEnclosingInterval() !=
      //        stencilAxis; if(!MergeDoMethodsOfStage && !MergeDoMethodsOfStencil) {
      //          continue;
      //        }
      //
      //        // Try to merge each Do-Method of our `curStage`
      //        for(auto curDoMethodIt = curStage.childrenBegin();
      //            curDoMethodIt != curStage.childrenEnd();) {
      //          iir::DoMethod& curDoMethod = **curDoMethodIt;
      //
      //          bool mergedDoMethod = false;
      //
      //          // Start from the next stage and iterate upwards (i.e backwards) to find a
      //          suitable
      //          // stage to merge
      //          for(auto candidateStageIt = std::next(curStageIt);
      //              candidateStageIt != multiStage.childrenREnd(); ++candidateStageIt) {
      //            iir::Stage& candidateStage = **candidateStageIt;
      //
      //            // do the iterationspaces match?
      //            if(candidateStage.getIterationSpace() != curStage.getIterationSpace()) {
      //              continue;
      //            }
      //            // can only merge stages with same location type (for Cartesian they are both
      //            // std::nullopt)
      //            if(candidateStage.getLocationType() != curStage.getLocationType()) {
      //              continue;
      //            }
      //
      //            // Does the interval of `curDoMethod` overlap with any DoMethod interval in
      //            // `candidateStage`?
      //            auto candidateDoMethodIt = std::find_if(
      //                candidateStage.childrenBegin(), candidateStage.childrenEnd(),
      //                [&](const iir::Stage::DoMethodSmartPtr_t& doMethodPtr) {
      //                  return doMethodPtr->getInterval().overlaps(curDoMethod.getInterval());
      //                });
      //
      //            if(candidateDoMethodIt != candidateStage.childrenEnd()) {
      //
      //              // Check if our interval exists (Note that if we overlap with a DoMethod but
      //              our
      //              // interval does not exists, we cannot merge ourself into this stage).
      //              candidateDoMethodIt =
      //                  std::find_if(candidateStage.childrenBegin(), candidateStage.childrenEnd(),
      //                               [&](const iir::Stage::child_smartptr_t<iir::DoMethod>&
      //                               doMethodPtr) {
      //                                 return doMethodPtr->getInterval() ==
      //                                 curDoMethod.getInterval();
      //                               });
      //
      //              if(candidateDoMethodIt != candidateStage.childrenEnd()) {
      //
      //                // Check if we can append our `curDoMethod` to that `candidateDoMethod`. We
      //                need
      //                // to check if the resulting dep. graph is a DAG and does not contain any
      //                horizontal
      //                // dependencies
      //                iir::DoMethod& candidateDoMethod = **candidateDoMethodIt;
      //
      //                auto& candidateDepGraph = candidateDoMethod.getDependencyGraph();
      //                auto& curDepGraph = curDoMethod.getDependencyGraph();
      //
      //                auto newDepGraph =
      //                iir::DependencyGraphAccesses(stencilInstantiation->getMetaData(),
      //                                                                *candidateDepGraph,
      //                                                                *curDepGraph);
      //
      //                if(newDepGraph.isDAG() &&
      //                !hasHorizontalReadBeforeWriteConflict(newDepGraph)) {
      //
      //                  if(MergeStagesOfStencil) {
      //                    candidateStage.appendDoMethod(*curDoMethodIt, *candidateDoMethodIt,
      //                                                  std::move(newDepGraph));
      //                    for(auto& doMethod : candidateStage.getChildren()) {
      //                      doMethod->update(iir::NodeUpdateType::level);
      //                    }
      //                    candidateStage.update(iir::NodeUpdateType::level);
      //                    mergedDoMethod = true;
      //
      //                    // We moved one Do-Method away and thus broke our full axis
      //                    MergeDoMethodsOfStage = true;
      //                    break;
      //                  }
      //                }
      //              }
      //            } else {
      //              // Interval does not exists in `candidateStage`, just insert our DoMethod
      //              if(MergeDoMethodsOfStencil && MergeDoMethodsOfStage) {
      //                candidateStage.addDoMethod(std::move(*curDoMethodIt));
      //                // CARTO
      //                for(auto& doMethod : candidateStage.getChildren()) {
      //                  doMethod->update(iir::NodeUpdateType::level);
      //                }
      //                candidateStage.update(iir::NodeUpdateType::level);
      //                mergedDoMethod = true;
      //                break;
      //              }
      //            }
      //
      //            // The `curStage` depends on `candidateStage`, we thus cannot go further upwards
      //            if(stageDAG.depends(curStage.getStageID(), candidateStage.getStageID())) {
      //              break;
      //            }
      //          }
      //
      //          if(mergedDoMethod) {
      //            curDoMethodIt = curStage.childrenErase(curDoMethodIt);
      //            updateFields = true;
      //          } else
      //            curDoMethodIt++;
      //        }
      //
      //        if(updateFields) {
      //          for(auto& doMethod : curStage.getChildren()) {
      //            doMethod->update(iir::NodeUpdateType::level);
      //          }
      //
      //          curStage.update(iir::NodeUpdateType::level);
      //        }
      //        curStageIt++;
      //      }
      //
      //      // remote empty stages
      //      for(auto curStageIt = multiStage.childrenBegin(); curStageIt !=
      //      multiStage.childrenEnd();) {
      //        if((*curStageIt)->childrenEmpty()) {
      //          curStageIt = multiStage.childrenErase(curStageIt);
      //        } else
      //          curStageIt++;
      //      }
    }
  }

  return true;
}

bool PassMultiStageMerger::multiStageDependsOn(
    const std::unique_ptr<dawn::iir::MultiStage>& thisMS,
    const std::unique_ptr<dawn::iir::MultiStage>& otherMS,
    const dawn::iir::DependencyGraphStage& stageDAG) {
  for(const auto& thisStage : thisMS->getChildren()) {
    for(const auto& otherStage : otherMS->getChildren()) {
      if(stageDAG.depends(thisStage->getStageID(), otherStage->getStageID()))
        return true;
    }
  }
  return false;
}

} // namespace dawn
