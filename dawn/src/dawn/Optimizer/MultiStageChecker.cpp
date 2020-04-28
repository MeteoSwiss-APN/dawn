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
#include "dawn/Optimizer/MultiStageChecker.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Exception.h"

namespace dawn {
MultiStageChecker::MultiStageChecker(iir::StencilInstantiation* instantiation,
                                     const int maxHaloPoints)
    : instantiation_(instantiation), maxHaloPoints_(maxHaloPoints) {}

void MultiStageChecker::run() {
  unsigned nMultiStages = 0;
  for(const auto& multiStage : iterateIIROver<iir::MultiStage>(*instantiation_->getIIR())) {
    const auto& firstStage = *(multiStage->getChildren().begin());
    auto multiStageDependencyGraph =
        multiStage->getDependencyGraphOfInterval(firstStage->getEnclosingExtendedInterval());

    for(const auto& stage : multiStage->getChildren()) {
      // Merge stage into dependency graph
      const iir::DoMethod& doMethod = stage->getSingleDoMethod();
      multiStageDependencyGraph.merge(*doMethod.getDependencyGraph());
    }

    std::string errorMessage;
    if(!multiStageDependencyGraph.empty()) {
      if(!multiStageDependencyGraph.isDAG()) {
        errorMessage =
            "Multistage " + std::to_string(nMultiStages) + "has cycle in dependency graph";
      } else if(multiStageDependencyGraph.exceedsMaxBoundaryPoints(maxHaloPoints_)) {
        errorMessage = "Multistage " + std::to_string(nMultiStages) +
                       " extent exeeds max halo points " + std::to_string(maxHaloPoints_);
      }
    }

    if(!errorMessage.empty())
      throw CompileError(errorMessage);

    nMultiStages += 1;
  }
}

} // namespace dawn
