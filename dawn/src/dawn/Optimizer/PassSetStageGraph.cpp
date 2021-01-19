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

#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/STLExtras.h"

namespace dawn {

/// @brief Compute the dependency between the stages `from` and `to`
/// @return `true` if the stage `from` depends on `to`, `false` otherwise
static bool depends(const iir::Stage& fromStage, const iir::Stage& toStage) {
  if(!fromStage.overlaps(toStage))
    return false;

  for(const auto& fromFieldPair : fromStage.getFields()) {
    const iir::Field& fromField = fromFieldPair.second;
    for(const auto& toFieldPair : toStage.getFields()) {
      const iir::Field& toField = toFieldPair.second;
      if(fromField.getAccessID() != toField.getAccessID())
        continue;

      iir::Field::IntendKind fromFieldIntend = fromField.getIntend();
      iir::Field::IntendKind toFieldIntend = toField.getIntend();

      switch(fromFieldIntend) {
      case iir::Field::IntendKind::Output:
        // This used to check if IntendKind was Input or InputOutput, but this
        // reorders stages when there is a WAW dependency. Instead, we should
        // catch all output dependencies
        return true;
        break;
      case iir::Field::IntendKind::InputOutput:
        return true;
      case iir::Field::IntendKind::Input:
        if(toFieldIntend == iir::Field::IntendKind::Output ||
           toFieldIntend == iir::Field::IntendKind::InputOutput)
          return true;
        break;
      }
    }
  }
  return false;
}

bool PassSetStageGraph::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                            const Options& options) {
  int stencilIdx = 0;

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;
    int numStages = stencil.getNumStages();

    auto stageDAG = iir::DependencyGraphStage(stencilInstantiation);

    // Build DAG of stages (backward sweep)
    for(int i = numStages - 1; i >= 0; --i) {
      const auto& fromStagePtr = stencil.getStage(i);
      stageDAG.insertNode(fromStagePtr->getStageID());
      int curStageID = fromStagePtr->getStageID();

      for(int j = i - 1; j >= 0; --j) {
        const auto& toStagePtr = stencil.getStage(j);
        if(depends(*fromStagePtr, *toStagePtr))
          stageDAG.insertEdge(curStageID, toStagePtr->getStageID());
      }
    }

    if(options.DumpStageGraph)
      stageDAG.toDot("stage_" + stencilInstantiation->getName() + "_s" +
                     std::to_string(stencilIdx) + ".dot");

    stencil.setStageDependencyGraph(std::move(stageDAG));
  }

  return true;
}

} // namespace dawn
