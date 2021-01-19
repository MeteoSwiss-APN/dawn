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
#include "ControlFlowDescriptor.h"
#include "StencilMetaInformation.h"

namespace dawn {
namespace iir {

ControlFlowDescriptor ControlFlowDescriptor::clone() const {
  ControlFlowDescriptor copy;
  for(const auto& stmt : controlFlowStatements_) {
    copy.controlFlowStatements_.push_back(stmt->clone());
  }
  return copy;
}
void ControlFlowDescriptor::removeStencilCalls(const std::set<int>& stencilIDs,
                                               iir::StencilMetaInformation& metadata) {
  std::vector<StmtConstIterator> stmtsToRemove;
  for(StmtConstIterator it = getStatements().begin(); it != getStatements().end(); ++it) {
    const std::shared_ptr<ast::Stmt>& stmt = *it;
    if(isa<ast::StencilCallDeclStmt>(stmt.get())) {
      auto callDecl = std::static_pointer_cast<ast::StencilCallDeclStmt>(stmt);
      bool remove = false;
      for(int id : stencilIDs) {
        if(metadata.getStencilIDFromStencilCallStmt(callDecl) == id) {
          remove = true;
        }
      }
      if(remove) {
        stmtsToRemove.push_back(it);
      }
    }
  }
  for(StmtConstIterator it : stmtsToRemove) {
    eraseStmt(it);
  }
  for(auto stencilID : stencilIDs) {
    metadata.eraseStencilID(stencilID);
  }
}

} // namespace iir
} // namespace dawn
