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
#include "dawn/Support/Iterator.h"

namespace dawn {

std::unique_ptr<iir::Stencil>
ReorderStrategyGreedy::reorder(iir::StencilInstantiation* instantiation,
                               const std::unique_ptr<iir::Stencil>& stencil,
                               const Options& options) {

  auto& metadata = instantiation->getMetaData();
  std::unique_ptr<iir::Stencil> newStencil = std::make_unique<iir::Stencil>(
      metadata, stencil->getStencilAttributes(), stencil->getStencilID());
  auto const& stageDAG = *stencil->getStageDependencyGraph();
  newStencil->setStageDependencyGraph(iir::DependencyGraphStage(stageDAG));

  int totalNewStages = 0;
  for(auto [msIdx, multiStage] : enumerate(stencil->getChildren())) {
    iir::LoopOrderKind stageLoopOrder = multiStage->getLoopOrder();
    newStencil->insertChild(std::make_unique<iir::MultiStage>(metadata, stageLoopOrder));

    int newNumStages = 0;
    for(auto& stage : multiStage->getChildren()) {
      // Compute the best possible position to where we can move this stage without violating
      // any dependencies
      int stageIdx = newNumStages - 1;
      for(; stageIdx >= 0; --stageIdx) {
        int newStageID = newStencil->getStage(totalNewStages + stageIdx)->getStageID();
        if(stageDAG.depends(stage->getStageID(), newStageID))
          break;
      }

      iir::Stencil::StagePosition stagePos(msIdx, stageIdx);
      newStencil->insertStage(stagePos, std::move(stage));
      newNumStages += 1;
    }
    totalNewStages += newNumStages;
  }

  return newStencil;
}

} // namespace dawn

