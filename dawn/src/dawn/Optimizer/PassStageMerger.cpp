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

#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Support/FileSystem.h"

namespace dawn {

bool PassStageMerger::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                          const Options& options) {
  // Do we need to run this Pass?
  bool stencilNeedsMergePass = false;
  for(const auto& stencilPtr : stencilInstantiation->getStencils())
    stencilNeedsMergePass |= stencilPtr->getStencilAttributes().hasOneOf(
        sir::Attr::Kind::MergeStages, sir::Attr::Kind::MergeDoMethods);

  // ... Nope
  if(!options.MergeDoMethods && !stencilNeedsMergePass)
    return true;

  const std::string filenameWE =
      fs::path(stencilInstantiation->getMetaData().getFileName()).filename().stem();

  if(options.WriteStencilInstantiation)
    stencilInstantiation->jsonDump(filenameWE + "_before_stage_merger.json");

  for(const auto& stencil : stencilInstantiation->getStencils()) {
    if(stencil->isEmpty()) {
      continue;
    }
    const auto& stencilAxis = stencil->getAxis(false);
    const auto& stageDAG = *(stencil->getStageDependencyGraph());
    const auto& attributes = stencil->getStencilAttributes();

    // Do we need to run the analysis for this stencil?
    bool mergeDoMethodsOfStencil =
        attributes.has(sir::Attr::Kind::MergeDoMethods) || options.MergeDoMethods;

    // Note that the underlying assumption is that stages in the same multi-stage are guaranteed to
    // have no counter loop-oorder vertical dependencies. We can thus treat each multi-stage in
    // isolation!
    for(auto& multiStage : stencil->getChildren()) {
      // Iterate stages backwards (bottom -> top)
      for(auto curStageIt = multiStage->childrenRBegin();
          curStageIt != multiStage->childrenREnd();) {
        iir::Stage& curStage = **curStageIt;
        bool updateFields = false;

        // If our Do-Methods already spans the entire axis, we don't want to destroy that property
        bool mergeDoMethodsOfStage = curStage.getEnclosingInterval() != stencilAxis;

        // Try to merge each Do-Method of our `curStage`
        for(auto curDoMethodIt = curStage.childrenBegin();
            curDoMethodIt != curStage.childrenEnd();) {
          iir::DoMethod& curDoMethod = **curDoMethodIt;
          bool mergedDoMethod = false;

          // Start from the next stage and iterate upwards (i.e backwards) to find a suitable
          // stage to merge
          for(auto candidateStageIt = std::next(curStageIt);
              candidateStageIt != multiStage->childrenREnd(); ++candidateStageIt) {
            iir::Stage& candidateStage = **candidateStageIt;

            // do the iterationspaces match?
            if(candidateStage.getIterationSpace() != curStage.getIterationSpace())
              continue;

            // can only merge stages with same location type (for Cartesian they are both
            // std::nullopt)
            if(candidateStage.getLocationType() != curStage.getLocationType())
              continue;

            // Does the interval of `curDoMethod` overlap with any DoMethod interval in
            // `candidateStage`?
            auto candidateDoMethodIt = std::find_if(
                candidateStage.childrenBegin(), candidateStage.childrenEnd(),
                [&](const iir::Stage::DoMethodSmartPtr_t& doMethodPtr) {
                  return doMethodPtr->getInterval().overlaps(curDoMethod.getInterval());
                });

            if(candidateDoMethodIt != candidateStage.childrenEnd()) {
              // Check if our interval exists (Note that if we overlap with a DoMethod but our
              // interval does not exists, we cannot merge ourself into this stage).
              candidateDoMethodIt =
                  std::find_if(candidateStage.childrenBegin(), candidateStage.childrenEnd(),
                               [&](const iir::Stage::child_smartptr_t<iir::DoMethod>& doMethodPtr) {
                                 return doMethodPtr->getInterval() == curDoMethod.getInterval();
                               });

              if(candidateDoMethodIt != candidateStage.childrenEnd()) {
                // Check if we can append our `curDoMethod` to that `candidateDoMethod`. We need
                // to check if the resulting dep. graph is a DAG and does not contain any horizontal
                // dependencies
                iir::DoMethod& candidateDoMethod = **candidateDoMethodIt;

                auto& candidateDepGraph = candidateDoMethod.getDependencyGraph();
                auto& curDepGraph = curDoMethod.getDependencyGraph();

                auto newDepGraph = iir::DependencyGraphAccesses(stencilInstantiation->getMetaData(),
                                                                *candidateDepGraph, *curDepGraph);

                if(newDepGraph.isDAG() && !hasHorizontalReadBeforeWriteConflict(newDepGraph)) {
                  candidateStage.appendDoMethod(*curDoMethodIt, *candidateDoMethodIt,
                                                std::move(newDepGraph));
                  for(auto& doMethod : candidateStage.getChildren()) {
                    doMethod->update(iir::NodeUpdateType::level);
                  }
                  candidateStage.update(iir::NodeUpdateType::level);
                  mergedDoMethod = true;

                  // We moved one Do-Method away and thus broke our full axis
                  mergeDoMethodsOfStage = true;
                  break;
                }
              }
            } else {
              // Interval does not exists in `candidateStage`, just insert our DoMethod
              if(mergeDoMethodsOfStencil && mergeDoMethodsOfStage) {
                candidateStage.addDoMethod(std::move(*curDoMethodIt));
                // CARTO
                for(auto& doMethod : candidateStage.getChildren()) {
                  doMethod->update(iir::NodeUpdateType::level);
                }
                candidateStage.update(iir::NodeUpdateType::level);
                mergedDoMethod = true;
                break;
              }
            }

            // The `curStage` depends on `candidateStage`, we thus cannot go further upwards
            if(stageDAG.depends(curStage.getStageID(), candidateStage.getStageID()))
              break;
          }

          if(mergedDoMethod) {
            curDoMethodIt = curStage.childrenErase(curDoMethodIt);
            updateFields = true;
          } else {
            curDoMethodIt++;
          }
        }

        if(updateFields) {
          for(auto& doMethod : curStage.getChildren()) {
            doMethod->update(iir::NodeUpdateType::level);
          }
          curStage.update(iir::NodeUpdateType::level);
        }
        curStageIt++;
      }

      // Remove empty stages
      for(auto curStageIt = multiStage->childrenBegin(); curStageIt != multiStage->childrenEnd();) {
        if((*curStageIt)->childrenEmpty()) {
          curStageIt = multiStage->childrenErase(curStageIt);
        } else {
          curStageIt++;
        }
      }
    }
  }

  if(options.WriteStencilInstantiation)
    stencilInstantiation->jsonDump(filenameWE + "_after_stage_merger.json");

  return true;
}

} // namespace dawn
