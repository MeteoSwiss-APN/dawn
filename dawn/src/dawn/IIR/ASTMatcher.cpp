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
#include "dawn/IIR/ASTMatcher.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/AST/ASTVisitor.h"

namespace dawn {
namespace iir {

ASTMatcher::ASTMatcher(iir::StencilInstantiation* instantiation)
    : instantiation_(instantiation), metadata_(instantiation->getMetaData()) {}

std::vector<std::shared_ptr<ast::Stmt>>& ASTMatcher::match(ast::Stmt::Kind kind) {
  stmtMatches_.clear();
  stmtKind_ = kind;
  iterate(instantiation_);
  return stmtMatches_;
}

std::vector<std::shared_ptr<ast::Expr>>& ASTMatcher::match(ast::Expr::Kind kind) {
  exprMatches_.clear();
  exprKind_ = kind;
  stmtKind_ = ast::Stmt::Kind::ExprStmt;
  iterate(instantiation_);
  return exprMatches_;
}

void ASTMatcher::iterate(iir::StencilInstantiation* instantiation) {
  // Traverse stencil statements
  for(const auto& stencil : instantiation->getStencils()) {
    iterate(stencil);
  }
  // Traverse statements outside of stencils, e.g., in the 'run' method
  for(const auto& statement : instantiation->getIIR()->getControlFlowDescriptor().getStatements()) {
    statement->accept(*this);
  }
}

void ASTMatcher::iterate(const std::unique_ptr<iir::Stencil>& stencil) {
  for(const auto& multiStage : stencil->getChildren()) {
    iterate(multiStage);
  }
}

void ASTMatcher::iterate(const std::unique_ptr<iir::MultiStage>& multiStage) {
  for(const auto& stage : multiStage->getChildren()) {
    iterate(stage);
  }
}

void ASTMatcher::iterate(const std::unique_ptr<iir::Stage>& stage) {
  for(const auto& doMethod : stage->getChildren()) {
    iterate(doMethod);
  }
}

void ASTMatcher::iterate(const std::unique_ptr<iir::DoMethod>& doMethod) {
  for(const auto& statement : doMethod->getAST().getStatements()) {
    statement->accept(*this);
  }
}

void ASTMatcher::check(const std::shared_ptr<ast::Stmt>& stmt) {
  if(stmtKind_ == stmt->getKind())
    stmtMatches_.push_back(stmt);
}

void ASTMatcher::check(const std::shared_ptr<ast::Expr>& expr) {
  if(exprKind_ == expr->getKind())
    exprMatches_.push_back(expr);
}

void ASTMatcher::visit(const std::shared_ptr<ast::BlockStmt>& stmt) {
  check(stmt);
  for(const auto& child : stmt->getStatements()) {
    child->accept(*this);
  }
}

void ASTMatcher::visit(const std::shared_ptr<ast::ExprStmt>& stmt) {
  check(stmt);
  stmt->getExpr()->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::ReturnStmt>& stmt) {
  check(stmt);
  stmt->getExpr()->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::IfStmt>& stmt) {
  check(stmt);
  stmt->getCondExpr()->accept(*this);
  stmt->getThenStmt()->accept(*this);
  if(stmt->hasElse())
    stmt->getElseStmt()->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::VarDeclStmt>& stmt) {
  check(stmt);
  for(const auto& expr : stmt->getInitList())
    expr->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::AssignmentExpr>& expr) {
  check(expr);
  for(auto& child : expr->getChildren())
    child->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) {
  check(expr);
  for(auto& child : expr->getChildren())
    child->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::UnaryOperator>& expr) {
  check(expr);
  for(auto& child : expr->getChildren())
    child->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::BinaryOperator>& expr) {
  check(expr);
  for(auto& child : expr->getChildren())
    child->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::TernaryOperator>& expr) {
  check(expr);
  for(auto& child : expr->getChildren())
    child->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::FunCallExpr>& expr) {
  check(expr);
  for(auto& child : expr->getChildren())
    child->accept(*this);
}

void ASTMatcher::visit(const std::shared_ptr<ast::StencilFunCallExpr>& expr) { check(expr); }

void ASTMatcher::visit(const std::shared_ptr<ast::StencilFunArgExpr>& expr) { check(expr); }

void ASTMatcher::visit(const std::shared_ptr<ast::VarAccessExpr>& expr) { check(expr); }

void ASTMatcher::visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) {
  check(expr);
  ASTVisitorForwardingNonConst::visit(expr);
}

void ASTMatcher::visit(const std::shared_ptr<ast::LiteralAccessExpr>& expr) { check(expr); }

} // namespace iir
} // namespace dawn
