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

#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"

namespace dawn {
namespace sir {

//===------------------------------------------------------------------------------------------===//
//     VerticalRegionDeclStmt
//===------------------------------------------------------------------------------------------===//

VerticalRegionDeclStmt::VerticalRegionDeclStmt(
    const std::shared_ptr<AST>& ast, const std::shared_ptr<VerticalRegion>& verticalRegion,
    SourceLocation loc)
    : SIRASTData::StmtData(), SIRASTData::VerticalRegionDeclStmt(verticalRegion),
      Stmt(SK_VerticalRegionDeclStmt, loc), ast_(ast) {}

VerticalRegionDeclStmt::VerticalRegionDeclStmt(const VerticalRegionDeclStmt& stmt)
    : SIRASTData::StmtData(), SIRASTData::VerticalRegionDeclStmt(stmt),
      Stmt(SK_VerticalRegionDeclStmt, stmt.getSourceLocation()), ast_(stmt.getAST()->clone()) {}

VerticalRegionDeclStmt& VerticalRegionDeclStmt::operator=(VerticalRegionDeclStmt stmt) {
  SIRASTData::VerticalRegionDeclStmt::operator=(stmt);
  assign(stmt);
  ast_ = std::move(stmt.getAST());
  return *this;
}

VerticalRegionDeclStmt::~VerticalRegionDeclStmt() {}

std::shared_ptr<Stmt> VerticalRegionDeclStmt::clone() const {
  return std::make_shared<VerticalRegionDeclStmt>(*this);
}

bool VerticalRegionDeclStmt::equals(const Stmt* other) const {
  const VerticalRegionDeclStmt* otherPtr = dyn_cast<VerticalRegionDeclStmt>(other);
  return otherPtr && Stmt::equals(other) && compareAst(ast_, otherPtr->getAST()).second;
}

// TODO: ast::ASTVisitor cannot be extended to Stmts introduced outside of component AST (such as
// sir::VerticalRegionDeclStmt), except by subclassing it. But this requires the following dynamic
// casts as ast::ASTVisitor doesn't have a method visit(sir::VerticalRegionDeclStmt).
// This solution, however, is not compile-time safe.

void VerticalRegionDeclStmt::accept(ast::ASTVisitor<SIRASTData>& visitor) {
  if(ASTVisitor* sirVisitor = dynamic_cast<ASTVisitor*>(&visitor))
    sirVisitor->visit(std::static_pointer_cast<VerticalRegionDeclStmt>(shared_from_this()));
  else
    throw std::runtime_error("Invalid ASTVisitor used on VerticalRegionDeclStmt.");
}

void VerticalRegionDeclStmt::accept(ast::ASTVisitorNonConst<SIRASTData>& visitor) {
  if(ASTVisitorNonConst* sirVisitor = dynamic_cast<ASTVisitorNonConst*>(&visitor))
    sirVisitor->visit(std::static_pointer_cast<VerticalRegionDeclStmt>(shared_from_this()));
  else
    throw std::runtime_error("Invalid ASTVisitorNonConst used on VerticalRegionDeclStmt.");
}

void VerticalRegionDeclStmt::accept(ast::ASTVisitorForwarding<SIRASTData>& visitor) {
  if(ASTVisitorForwarding* sirVisitor = dynamic_cast<ASTVisitorForwarding*>(&visitor))
    sirVisitor->visit(std::static_pointer_cast<VerticalRegionDeclStmt>(shared_from_this()));
  else
    throw std::runtime_error("Invalid ASTVisitorForwarding used on VerticalRegionDeclStmt.");
}

std::shared_ptr<Stmt>
VerticalRegionDeclStmt::acceptAndReplace(ast::ASTVisitorPostOrder<SIRASTData>& visitor) {
  if(ASTVisitorPostOrder* sirVisitor = dynamic_cast<ASTVisitorPostOrder*>(&visitor))
    return sirVisitor->visitAndReplace(
        std::static_pointer_cast<VerticalRegionDeclStmt>(shared_from_this()));
  else
    throw std::runtime_error("Invalid ASTVisitorPostOrder used on VerticalRegionDeclStmt.");
}

void VerticalRegionDeclStmt::accept(ast::ASTVisitorForwardingNonConst<SIRASTData>& visitor) {
  if(ASTVisitorForwardingNonConst* sirVisitor =
         dynamic_cast<ASTVisitorForwardingNonConst*>(&visitor))
    sirVisitor->visit(std::static_pointer_cast<VerticalRegionDeclStmt>(shared_from_this()));
  else
    throw std::runtime_error(
        "Invalid ASTVisitorForwardingNonConst used on VerticalRegionDeclStmt.");
}

void VerticalRegionDeclStmt::accept(ast::ASTVisitorDisabled<SIRASTData>& visitor) {
  if(ASTVisitorDisabled* sirVisitor = dynamic_cast<ASTVisitorDisabled*>(&visitor))
    sirVisitor->visit(std::static_pointer_cast<VerticalRegionDeclStmt>(shared_from_this()));
  else
    throw std::runtime_error("Invalid ASTVisitorDisabled used on VerticalRegionDeclStmt.");
}

} // namespace sir
} // namespace dawn
