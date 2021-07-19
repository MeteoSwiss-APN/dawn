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
  // everything can be assigned to a full field.
  // dimensionless expressions are ok (-1).
  // otherwise dimensions need to match (this prohibits vert = hor/full and hor = vert)
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

void IntegrityChecker::visit(const std::shared_ptr<ast::VarAccessExpr>& expr) {
  int accessID = iir::getAccessID(expr);
  auto& idToDim = metadata_.getFieldIDToDimsMap();
  if(!idToDim.count(accessID)) {
    curDimensions_ = -1;
  } else {
    curDimensions_ = idToDim.at(accessID).numSpatialDimensions();
  }
}

void IntegrityChecker::visit(const std::shared_ptr<ast::AssignmentExpr>& expr) {
  std::shared_ptr<ast::Expr>& left = expr->getLeft();
  // Check whether literal expressions are being assigned
  if(ast::LiteralAccessExpr::classof(left.get())) {
    std::string value = dyn_cast<ast::LiteralAccessExpr>(left.get())->getValue();
    throw SemanticError("Attempt to assign constant expression " + value, metadata_.getFileName(),
                        expr->getSourceLocation().Line);
  }

  if(ast::FieldAccessExpr::classof(left.get())) {
    if(dyn_cast<ast::FieldAccessExpr>(left.get())->getOffset().verticalShift() != 0) {
      throw SemanticError("Attempt to write vertically offset ", metadata_.getFileName(),
                          expr->getSourceLocation().Line);
    }
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

  // we leave the strucutred world alone for now
  if(instantiation_->getIIR()->getGridType() == ast::GridType::Unstructured &&
     !dimensionsCompatible(leftDim, rightDim)) {
    throw SemanticError("trying to assign " + dimToStr(leftDim) + "d field to " +
                            dimToStr(rightDim) + "d field!",
                        metadata_.getFileName(), expr->getSourceLocation().Line);
  }

  curDimensions_ = oldDim;
} // namespace dawn

void IntegrityChecker::visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) {

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

  auto dim = metadata_.getFieldDimensions(metadata_.getNameToAccessIDMap().at(expr->getName()));
  if(!dim.K() &&
     (expr->getOffset().hasVerticalIndirection() || expr->getOffset().verticalShift() != 0)) {
    throw SemanticError("Attempting to read with vertical offset from horizontal field!");
  }

  ast::ASTVisitorForwardingNonConst::visit(expr);

  curDimensions_ = metadata_.getFieldDimensions(accessID).numSpatialDimensions();
}

void IntegrityChecker::visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) {
  bool parentHadIterationContext = parentHasIterationContext_;
  parentHasIterationContext_ = true;
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
  parentHasIterationContext_ = parentHadIterationContext;
}

void IntegrityChecker::visit(const std::shared_ptr<ast::LoopStmt>& expr) {
  parentHasIterationContext_ = true;
  for(auto& stmt : expr->getChildren())
    stmt->accept(*this);
  parentHasIterationContext_ = false;
}

} // namespace dawn
