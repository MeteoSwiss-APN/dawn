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
#include "dawn/Validator/IntegrityChecker.h"
#include "dawn/AST/GridType.h"
#include "dawn/IIR/ASTExpr.h"

static int dimensionsCompatible(int leftDim, int rightDim) {
  return leftDim == 3 || rightDim == -1 || leftDim == -1 || leftDim == rightDim;
}

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

void IntegrityChecker::visit(const std::shared_ptr<iir::AssignmentExpr>& expr) {
  std::shared_ptr<iir::Expr>& left = expr->getLeft();
  // Check whether literal expressions are being assigned
  if(iir::LiteralAccessExpr::classof(left.get())) {
    std::string value = dyn_cast<iir::LiteralAccessExpr>(left.get())->getValue();
    throw SemanticError("Attempt to assign constant expression " + value, metadata_.getFileName(),
                        expr->getSourceLocation().Line);
  }

  int oldDim = curDimensions_;
  expr->getLeft()->accept(*this);
  int leftDim = curDimensions_;
  expr->getRight()->accept(*this);
  int rightDim = curDimensions_;

  auto dimToStr = [](int d) -> std::string {
    switch(d) {
    case 1:
      return "vertical";
    case 2:
      return "horizontal";
    case 3:
      return "full";
    default:
      return "";
    }
  };

  // we leave the unstrucutred world alone for now
  if(instantiation_->getIIR()->getGridType() == ast::GridType::Unstructured &&
     !dimensionsCompatible(leftDim, rightDim)) {
    throw SemanticError("trying to assign " + dimToStr(leftDim) + "d field to " +
                        dimToStr(rightDim) + "d field!");
  }

  curDimensions_ = oldDim;
}

void IntegrityChecker::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  int accessID = iir::getAccessID(expr);
  DAWN_ASSERT_MSG(metadata_.getFieldNameFromAccessID(accessID) == expr->getName(),
                  (std::string("Field name (") + std::string(expr->getName()) +
                   std::string(") doesn't match name registered in metadata(") +
                   std::string(metadata_.getFieldNameFromAccessID(accessID)) + std::string(")."))
                      .c_str());

  if(expr->getOffset().horizontalOffset().hasType() &&
     expr->getOffset().horizontalOffset().getGridType() == ast::GridType::Unstructured) {
    auto unstrOffset =
        ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset());
    if(unstrOffset.hasOffset() && !parentHasIterationContext_) {
      throw SemanticError("Attempting to offset read from/write to unstructured field outside of a "
                          "reduction expression or loop statement!");
    }
  }

  curDimensions_ = metadata_.getFieldDimensions(accessID).numSpatialDimensions();
  ast::ASTVisitorForwarding::visit(expr);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::UnaryOperator>& expr) {
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
}

void IntegrityChecker::visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) {
  bool parentHadIterationContext = parentHasIterationContext_;
  parentHasIterationContext_ = true;
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
  parentHasIterationContext_ = parentHadIterationContext;
}

void IntegrityChecker::visit(const std::shared_ptr<iir::LoopStmt>& expr) {
  parentHasIterationContext_ = true;
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
  parentHasIterationContext_ = false;
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

} // namespace dawn
