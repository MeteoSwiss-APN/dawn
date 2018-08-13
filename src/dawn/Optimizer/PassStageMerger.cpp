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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/FileUtil.h"

namespace dawn {

PassStageMerger::PassStageMerger() : Pass("PassStageMerger") {
  dependencies_.push_back("PassSetStageGraph");
}

bool PassStageMerger::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  // Do we need to run this Pass?
  bool stencilNeedsMergePass = false;
  for(const auto& stencilPtr : stencilInstantiation->getStencils())
    stencilNeedsMergePass |= stencilPtr->getSIRStencil()->Attributes.hasOneOf(
        sir::Attr::AK_MergeStages, sir::Attr::AK_MergeDoMethods);

  bool MergeStages = context->getOptions().MergeStages;
  bool MergeDoMethods = context->getOptions().MergeDoMethods;

  // ... Nope
  if(!MergeStages && !MergeDoMethods && !stencilNeedsMergePass)
    return true;

  std::string filenameWE = getFilenameWithoutExtension(context->getSIR()->Filename);
  if(context->getOptions().ReportPassStageMerger)
    stencilInstantiation->dumpAsJson(filenameWE + "_before.json", getName());

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;

    // Do we need to run the analysis for thist stencil?
    const std::shared_ptr<iir::DependencyGraphStage>& stageDAG = stencil.getStageDependencyGraph();
    bool MergeDoMethodsOfStencil =
        stencil.getSIRStencil()->Attributes.has(sir::Attr::AK_MergeDoMethods) || MergeDoMethods;
    bool MergeStagesOfStencil =
        stencil.getSIRStencil()->Attributes.has(sir::Attr::AK_MergeStages) || MergeStages;

    // Nope
    if(!MergeStagesOfStencil && !MergeDoMethodsOfStencil)
      continue;

    // Note that the underyling assumption is that stages in the same multi-stage are guaranteed to
    // have no counter loop-oorder vertical dependencies. We can thus treat each multi-stage in
    // isolation!
    for(const auto& multiStagePtr : stencil.getChildren()) {
      iir::MultiStage& multiStage = *multiStagePtr;

      // Iterate stages backwards (bottom -> top)
      for(auto curStageIt = multiStage.childrenRBegin(); curStageIt != multiStage.childrenREnd();) {

        iir::Stage& curStage = **curStageIt;

        bool updateFields = false;

        // If our Do-Methods already spans the entire axis, we don't want to destroy that property
        bool MergeDoMethodsOfStage = curStage.getEnclosingInterval() != stencil.getAxis(false);
        if(!MergeDoMethodsOfStage && !MergeDoMethodsOfStencil) {
          continue;
        }

        // Try to merge each Do-Method of our `curStage`
        for(auto curDoMethodIt = curStage.childrenBegin();
            curDoMethodIt != curStage.childrenEnd();) {
          iir::DoMethod& curDoMethod = **curDoMethodIt;

          bool mergedDoMethod = false;

          // Start from the next stage and iterate upwards (i.e backwards) to find a suitable
          // stage to merge
          for(auto candidateStageIt = std::next(curStageIt);
              candidateStageIt != multiStage.childrenREnd(); ++candidateStageIt) {
            iir::Stage& candidateStage = **candidateStageIt;
            //            std::vector<std::unique_ptr<iir::DoMethod>>& candiateDoMethods =
            //                candidateStage.getDoMethods();

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

                // Check if we can append our `curDoMethod` to that `candiateDoMethod`. We need
                // to check if the resulting dep. graph is a DAG and does not contain any horizontal
                // dependencies
                iir::DoMethod& candiateDoMethod = **candidateDoMethodIt;

                auto& candiateDepGraph = candiateDoMethod.getDependencyGraph();
                auto& curDepGraph = curDoMethod.getDependencyGraph();

                auto newDepGraph = std::make_shared<iir::DependencyGraphAccesses>(
                    stencilInstantiation.get(), candiateDepGraph, curDepGraph);

                if(newDepGraph->isDAG() &&
                   !hasHorizontalReadBeforeWriteConflict(newDepGraph.get())) {

                  if(MergeStagesOfStencil) {
                    candidateStage.appendDoMethod(*curDoMethodIt, *candidateDoMethodIt,
                                                  newDepGraph);
                    mergedDoMethod = true;

                    // We moved one Do-Method away and thus broke our full axis
                    MergeDoMethodsOfStage = true;
                    break;
                  }
                }
              }
            } else {
              // Interval does not exists in `candidateStage`, just insert our DoMethod
              if(MergeDoMethodsOfStencil && MergeDoMethodsOfStage) {
                candidateStage.addDoMethod(*curDoMethodIt);
                mergedDoMethod = true;
                break;
              }
            }

            // The `curStage` depends on `candidateStage`, we thus cannot go further upwards
            if(stageDAG->depends(curStage.getStageID(), candidateStage.getStageID())) {
              break;
            }
          }

          if(mergedDoMethod) {
            curDoMethodIt = curStage.childrenErase(curDoMethodIt);
            updateFields = true;
          } else
            curDoMethodIt++;
        }

        if(updateFields)
          curStage.update();
        curStageIt++;
      }

      // remote empty stages
      for(auto curStageIt = multiStage.childrenBegin(); curStageIt != multiStage.childrenEnd();) {
        if((*curStageIt)->childrenEmpty()) {
          curStageIt = multiStage.childrenErase(curStageIt);
        } else
          curStageIt++;
      }
    }
  }

  if(context->getOptions().ReportPassStageMerger)
    stencilInstantiation->dumpAsJson(filenameWE + "_after.json", getName());

  return true;
}

} // namespace dawn
