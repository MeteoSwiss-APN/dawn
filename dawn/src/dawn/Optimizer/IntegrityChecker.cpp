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
#include "dawn/Optimizer/IntegrityChecker.h"

namespace dawn {
IntegrityChecker::IntegrityChecker(iir::StencilInstantiation* instantiation)
    : instantiation_(instantiation), metadata_(instantiation->getMetaData()) {}

void IntegrityChecker::run() { iterate(instantiation_); }

void IntegrityChecker::iterate(iir::StencilInstantiation* instantiation) {
  // Traverse stencil statements
  for(const auto& stencil : instantiation->getStencils()) {
    iterate(stencil);
  }
  // Traverse statements outside of stencils, e.g., in the 'run' method
  for(const auto& statement : instantiation->getIIR()->getControlFlowDescriptor().getStatements()) {
    statement->accept(*this);
  }
}

void IntegrityChecker::iterate(const std::unique_ptr<iir::Stencil>& stencil) {
  for(const auto& multiStage : stencil->getChildren()) {
    iterate(multiStage);
  }
}

void IntegrityChecker::iterate(const std::unique_ptr<iir::MultiStage>& multiStage) {
  for(const auto& stage : multiStage->getChildren()) {
    iterate(stage);
  }
}

void IntegrityChecker::iterate(const std::unique_ptr<iir::Stage>& stage) {
  for(const auto& doMethod : stage->getChildren()) {
    iterate(doMethod);
  }
}

void IntegrityChecker::iterate(const std::unique_ptr<iir::DoMethod>& doMethod) {
  for(const auto& statement : doMethod->getAST().getStatements()) {
    statement->accept(*this);
  }
}

void IntegrityChecker::visit(const std::shared_ptr<iir::BlockStmt>& statement) {
  for(const auto& stmt : statement->getStatements()) {
    stmt->accept(*this);
  }
}

void IntegrityChecker::visit(const std::shared_ptr<iir::ExprStmt>& statement) {
  statement->getExpr()->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
  stmt->getExpr()->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::IfStmt>& stmt) {
  stmt->getCondExpr()->accept(*this);
  stmt->getThenStmt()->accept(*this);
  if(stmt->hasElse())
    stmt->getElseStmt()->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) {
  for(const auto& expr : stmt->getInitList())
    expr->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {}

void IntegrityChecker::visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {}

void IntegrityChecker::visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {}

void IntegrityChecker::visit(const std::shared_ptr<iir::AssignmentExpr>& expr) {
  std::shared_ptr<iir::Expr>& left = expr->getLeft();
  // Check whether literal expressions are being assigned
  if(iir::LiteralAccessExpr::classof(left.get())) {
    std::string value = dyn_cast<iir::LiteralAccessExpr>(left.get())->getValue();
    throw SemanticError("Attempt to assign constant expression " + value, metadata_.getFileName(),
                        expr->getSourceLocation().Line);
  }
}

void IntegrityChecker::visit(const std::shared_ptr<iir::UnaryOperator>& expr) {
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) {
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::BinaryOperator>& expr) {
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::TernaryOperator>& expr) {
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::FunCallExpr>& expr) {
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {}

void IntegrityChecker::visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {}

void IntegrityChecker::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {}

void IntegrityChecker::visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) {}

void IntegrityChecker::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {}

} // namespace dawn
