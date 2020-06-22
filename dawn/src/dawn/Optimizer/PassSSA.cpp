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
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/CreateVersionAndRename.h"
#include <unordered_set>

namespace dawn {

bool PassSSA::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                  const Options& options) {

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;

    iir::DependencyGraphAccesses DAG(stencilInstantiation->getMetaData());

    std::unordered_set<int> tochedAccessIDs;

    // Iterate each statement of the stencil (top -> bottom)
    for(int stageIdx = 0; stageIdx < stencil.getNumStages(); ++stageIdx) {
      const std::unique_ptr<iir::Stage>& stagePtr = stencil.getStage(stageIdx);

      iir::DoMethod& doMethod = stagePtr->getSingleDoMethod();
      for(int stmtIdx = 0; stmtIdx < doMethod.getAST().getStatements().size(); ++stmtIdx) {

        const std::shared_ptr<iir::Stmt>& stmt = doMethod.getAST().getStatements()[stmtIdx];

        iir::AssignmentExpr* assignment = nullptr;
        if(iir::ExprStmt* exprStmt = dyn_cast<iir::ExprStmt>(stmt.get()))
          assignment = dyn_cast<iir::AssignmentExpr>(exprStmt->getExpr().get());
        if(assignment) {
          std::vector<int> AccessIDsToRename;
          const auto& callerAccesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;

          for(const std::pair<int, iir::Extents>& readAccess : callerAccesses->getReadAccesses()) {
            int AccessID = readAccess.first;
            if(!tochedAccessIDs.count(AccessID))
              tochedAccessIDs.insert(AccessID);
          }

          // Every write to a field which was has been touched (read/written) will get a new version
          for(const std::pair<int, iir::Extents>& writeAccess :
              callerAccesses->getWriteAccesses()) {

            int AccessID = writeAccess.first;

            if(tochedAccessIDs.count(AccessID))
              // Field was already written to, rename it in all remaining occurences
              AccessIDsToRename.push_back(AccessID);

            tochedAccessIDs.insert(AccessID);
          }

          for(int AccessID : AccessIDsToRename) {
            tochedAccessIDs.insert(
                createVersionAndRename(stencilInstantiation.get(), AccessID, &stencil, stageIdx,
                                       stmtIdx, assignment->getLeft(), RenameDirection::Below));
          }

          DAG.insertStatement(stmt);
        }
      }
    }
  }
  return true;
}

} // namespace dawn
