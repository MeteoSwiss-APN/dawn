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

#include "dawn/Optimizer/PassSSA.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/IIR/StencilInstantiation.h"
#include <unordered_set>

namespace dawn {

PassSSA::PassSSA() : Pass("PassSSA") {}

bool PassSSA::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  if(!context->getOptions().SSA)
    return true;

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;

    std::shared_ptr<iir::DependencyGraphAccesses> DAG =
        std::make_shared<iir::DependencyGraphAccesses>(stencilInstantiation.get());

    std::unordered_set<int> tochedAccessIDs;

    // Iterate each statement of the stencil (top -> bottom)
    for(int stageIdx = 0; stageIdx < stencil.getNumStages(); ++stageIdx) {
      const std::unique_ptr<iir::Stage>& stagePtr = stencil.getStage(stageIdx);

      iir::DoMethod& doMethod = stagePtr->getSingleDoMethod();
      for(int stmtIdx = 0; stmtIdx < doMethod.getChildren().size(); ++stmtIdx) {

        const std::unique_ptr<iir::StatementAccessesPair>& stmtAccessesPair =
            doMethod.getChildren()[stmtIdx];

        AssignmentExpr* assignment = nullptr;
        if(ExprStmt* stmt = dyn_cast<ExprStmt>(stmtAccessesPair->getStatement()->ASTStmt.get()))
          assignment = dyn_cast<AssignmentExpr>(stmt->getExpr().get());

        std::vector<int> AccessIDsToRename;

        for(const std::pair<int, iir::Extents>& readAccess :
            stmtAccessesPair->getAccesses()->getReadAccesses()) {
          int AccessID = readAccess.first;
          if(!tochedAccessIDs.count(AccessID))
            tochedAccessIDs.insert(AccessID);
        }

        // Every write to a field which was has been touched (read/written) will get a new version
        for(const std::pair<int, iir::Extents>& writeAccess :
            stmtAccessesPair->getAccesses()->getWriteAccesses()) {

          int AccessID = writeAccess.first;

          if(tochedAccessIDs.count(AccessID))
            // Field was already written to, rename it in all remaining occurences
            AccessIDsToRename.push_back(AccessID);

          tochedAccessIDs.insert(AccessID);
        }

        for(int AccessID : AccessIDsToRename)
          tochedAccessIDs.insert(stencilInstantiation->createVersionAndRename(
              AccessID, &stencil, stageIdx, stmtIdx, assignment->getLeft(),
              iir::StencilInstantiation::RD_Below));

        DAG->insertStatementAccessesPair(stmtAccessesPair);
      }
    }
  }
  return true;
}

} // namespace dawn
