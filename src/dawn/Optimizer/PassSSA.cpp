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
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include <unordered_set>

namespace dawn {

PassSSA::PassSSA() : Pass("PassSSA") {}

bool PassSSA::run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  if(!context->getOptions().SSA)
    return true;

  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;

    std::shared_ptr<DependencyGraphAccesses> DAG =
        std::make_shared<DependencyGraphAccesses>(stencilInstantiation.get());

    std::unordered_set<int> tochedAccessIDs;

    // Iterate each statement of the stencil (top -> bottom)
    for(int stageIdx = 0; stageIdx < stencil.getNumStages(); ++stageIdx) {
      std::shared_ptr<Stage> stagePtr = stencil.getStage(stageIdx);

      DoMethod& doMethod = stagePtr->getSingleDoMethod();
      for(int stmtIdx = 0; stmtIdx < doMethod.getStatementAccessesPairs().size(); ++stmtIdx) {

        std::shared_ptr<StatementAccessesPair> stmtAccessesPair =
            doMethod.getStatementAccessesPairs()[stmtIdx];

        AssignmentExpr* assignment = nullptr;
        if(ExprStmt* stmt = dyn_cast<ExprStmt>(stmtAccessesPair->getStatement()->ASTStmt.get()))
          assignment = dyn_cast<AssignmentExpr>(stmt->getExpr().get());

        std::vector<int> AccessIDsToRename;

        for(const std::pair<int, Extents>& readAccess :
            stmtAccessesPair->getAccesses()->getReadAccesses()) {
          int AccessID = readAccess.first;
          if(!tochedAccessIDs.count(AccessID))
            tochedAccessIDs.insert(AccessID);
        }

        // Every write to a field which was has been touched (read/written) will get a new version
        for(const std::pair<int, Extents>& writeAccess :
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
              StencilInstantiation::RD_Below));

        DAG->insertStatementAccessesPair(stmtAccessesPair);
      }
    }
  }
  return true;
}

} // namespace dawn
