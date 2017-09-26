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

#include "gsl/Optimizer/PassStageMerger.h"
#include "gsl/Optimizer/DependencyGraphAccesses.h"
#include "gsl/Optimizer/DependencyGraphStage.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/Optimizer/ReadBeforeWriteConflict.h"
#include "gsl/Optimizer/StencilInstantiation.h"
#include "gsl/Support/FileUtil.h"

namespace gsl {

PassStageMerger::PassStageMerger() : Pass("PassStageMerger") {
  dependencies_.push_back("PassSetStageGraph");
}

bool PassStageMerger::run(StencilInstantiation* stencilInstantiation) {
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
    Stencil& stencil = *stencilPtr;

    // Do we need to run the analysis for thist stencil?
    const std::shared_ptr<DependencyGraphStage>& stageDAG = stencil.getStageDependencyGraph();
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
    for(const auto& multiStagePtr : stencil.getMultiStages()) {
      MultiStage& multiStage = *multiStagePtr;

      // Iterate stages backwards (bottom -> top)
      for(auto curStageIt = multiStage.getStages().rbegin();
          curStageIt != multiStage.getStages().rend();) {

        Stage& curStage = **curStageIt;
        std::vector<std::unique_ptr<DoMethod>>& curDoMethods = curStage.getDoMethods();

        bool updateFields = false;

        // If our Do-Methods already spans the entire axis, we don't want to destroy that property
        bool MergeDoMethodsOfStage = curStage.getEnclosingInterval() != stencil.getAxis(false);
        if(!MergeDoMethodsOfStage && !MergeDoMethodsOfStencil) {
          continue;
        }

        // Try to merge each Do-Method of our `curStage`
        for(auto curDoMethodIt = curDoMethods.begin(); curDoMethodIt != curDoMethods.end();) {
          DoMethod& curDoMethod = **curDoMethodIt;

          bool mergedDoMethod = false;

          // Start from the next stage and iterate upwards (i.e backwards) to find a suitable
          // stage to merge
          for(auto candiateStageIt = std::next(curStageIt);
              candiateStageIt != multiStage.getStages().rend(); ++candiateStageIt) {
            Stage& candiateStage = **candiateStageIt;
            std::vector<std::unique_ptr<DoMethod>>& candiateDoMethods =
                candiateStage.getDoMethods();

            // Does the interval of `curDoMethod` overlap with any DoMethod interval in
            // `candiateStage`?
            auto candiateDoMethodIt = std::find_if(
                candiateDoMethods.begin(), candiateDoMethods.end(),
                [&](const std::unique_ptr<DoMethod>& doMethodPtr) {
                  return doMethodPtr->getInterval().overlaps(curDoMethod.getInterval());
                });

            if(candiateDoMethodIt != candiateDoMethods.end()) {

              // Check if our interval exists (Note that if we overlap with a DoMethod but our
              // interval does not exists, we cannot merge ourself into this stage).
              auto candiateDoMethodIt =
                  std::find_if(candiateDoMethods.begin(), candiateDoMethods.end(),
                               [&](const std::unique_ptr<DoMethod>& doMethodPtr) {
                                 return doMethodPtr->getInterval() == curDoMethod.getInterval();
                               });

              if(candiateDoMethodIt != candiateDoMethods.end()) {

                // Check if we can append our `curDoMethod` to that `candiateDoMethod`. We need
                // to check if the resulting dep. graph is a DAG and does not contain any horizontal
                // dependencies
                DoMethod& candiateDoMethod = **candiateDoMethodIt;

                auto& candiateDepGraph = candiateDoMethod.getDependencyGraph();
                auto& curDepGraph = curDoMethod.getDependencyGraph();

                auto newDepGraph = std::make_shared<DependencyGraphAccesses>(
                    stencilInstantiation, candiateDepGraph, curDepGraph);

                if(newDepGraph->isDAG() &&
                   !hasHorizontalReadBeforeWriteConflict(newDepGraph.get())) {
                  if(MergeStagesOfStencil) {
                    candiateStage.appendDoMethod(*curDoMethodIt, *candiateDoMethodIt, newDepGraph);
                    mergedDoMethod = true;

                    // We moved one Do-Method away and thus broke our full axis
                    MergeDoMethodsOfStage = true;
                    break;
                  }
                }
              }
            } else {
              // Interval does not exists in `candiateStage`, just insert our DoMethod
              if(MergeDoMethodsOfStencil && MergeDoMethodsOfStage) {
                candiateStage.addDoMethod(*curDoMethodIt);
                mergedDoMethod = true;
                break;
              }
            }

            // The `curStage` depends on `candiateStage`, we thus cannot go further upwards
            if(stageDAG->depends(curStage.getStageID(), candiateStage.getStageID())) {
              break;
            }
          }

          if(mergedDoMethod) {
            curDoMethodIt = curDoMethods.erase(curDoMethodIt);
            updateFields = true;
          } else
            curDoMethodIt++;
        }

        // Stage is empty, remove it (the wirdness here stems from the fact that we have a reverse
        // iterator and erase expects a normal iterator ...)
        if(curDoMethods.empty())
          curStageIt = decltype(curStageIt)(multiStage.getStages().erase(--curStageIt.base()));
        else {
          if(updateFields)
            curStage.update();
          curStageIt++;
        }
      }
    }
  }

  if(context->getOptions().ReportPassStageMerger)
    stencilInstantiation->dumpAsJson(filenameWE + "_after.json", getName());

  return true;
}

} // namespace gsl
