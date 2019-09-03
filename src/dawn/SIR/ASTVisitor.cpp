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

#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/ASTStmt.h"

namespace dawn {
namespace sir {

void ASTVisitorForwarding::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  stmt->getAST()->accept(*this);
}

void ASTVisitorForwardingNonConst::visit(std::shared_ptr<VerticalRegionDeclStmt> stmt) {
  stmt->getAST()->accept(*this);
}

std::shared_ptr<Stmt>
ASTVisitorPostOrder::visitAndReplace(std::shared_ptr<VerticalRegionDeclStmt> const& stmt) {
  // TODO replace this as wel
  if(!preVisitNode(stmt))
    return stmt;
  auto repl = stmt->getAST()->acceptAndReplace(*this);
  if(repl && repl != stmt->getAST())
    stmt->getAST() = repl;
  return postVisitNode(stmt);
}

bool ASTVisitorPostOrder::preVisitNode(std::shared_ptr<VerticalRegionDeclStmt> const& stmt) {
  return true;
}

std::shared_ptr<Stmt>
ASTVisitorPostOrder::postVisitNode(std::shared_ptr<VerticalRegionDeclStmt> const& stmt) {
  return stmt;
}

void ASTVisitorDisabled::visit(const std::shared_ptr<VerticalRegionDeclStmt>& node) {
  DAWN_ASSERT_MSG(0, "Type not allowed in this context");
}

} // namespace sir
} // namespace dawn
