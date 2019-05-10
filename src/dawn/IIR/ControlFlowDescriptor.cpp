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

void ControlFlowDescriptor::insertStmt(std::shared_ptr<Statement>&& statment) {
  controlFlowStatements_.push_back(statment);
}

void ControlFlowDescriptor::removeStencilCalls(const std::set<int>& stencilIDs,
                                               iir::StencilMetaInformation& metadata) {
  for(auto it = getStatements().begin(); it != getStatements().end(); ++it) {
    std::shared_ptr<Stmt> stmt = (*it)->ASTStmt;
    if(isa<StencilCallDeclStmt>(stmt.get())) {
      auto callDecl = std::static_pointer_cast<StencilCallDeclStmt>(stmt);
      bool remove = false;
      for(int id : stencilIDs) {
        if(metadata.getStencilIDFromStencilCallStmt(callDecl) == id) {
          remove = true;
        }
      }
      if(remove) {
        it = eraseStmt(it);
      }
    }
  }
  for(auto stencilID : stencilIDs) {
    metadata.eraseStencilID(stencilID);
  }
}

} // namespace iir
} // namespace dawn
