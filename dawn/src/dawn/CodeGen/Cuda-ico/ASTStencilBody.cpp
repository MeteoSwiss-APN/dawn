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

#include "ASTStencilBody.h"
#include "dawn/AST/LocationType.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include <memory>
#include <optional>
#include <sstream>
#include <string>

namespace dawn {
namespace codegen {
namespace cudaico {

void ASTStencilBody::visit(const std::shared_ptr<ast::BlockStmt>& stmt) {
  indent_ += DAWN_PRINT_INDENT;
  auto indent = std::string(indent_, ' ');
  for(const auto& s : stmt->getStatements()) {
    ss_ << indent;
    s->accept(*this);
  }
  indent_ -= DAWN_PRINT_INDENT;
}

void ASTStencilBody::visit(const std::shared_ptr<ast::LoopStmt>& stmt) {
  const auto maybeChainPtr =
      dynamic_cast<const ast::ChainIterationDescr*>(stmt->getIterationDescrPtr());
  DAWN_ASSERT_MSG(maybeChainPtr, "general loop concept not implemented yet!\n");

  parentIsForLoop_ = true;
  ss_ << "for (int nbhIter = 0; nbhIter < "
      << chainToSparseSizeString(maybeChainPtr->getIterSpace()) << "; nbhIter++)";

  ss_ << "{\n";
  ss_ << "int nbhIdx = " << chainToTableString(maybeChainPtr->getIterSpace()) << "["
      << "pidx * " << chainToSparseSizeString(maybeChainPtr->getIterSpace()) << " + nbhIter"
      << "];\n";
  ss_ << "if (nbhIdx == DEVICE_MISSING_VALUE) { continue; }";

  stmt->getBlockStmt()->accept(*this);

  ss_ << "}\n";
  parentIsForLoop_ = false;
}
void ASTStencilBody::visit(const std::shared_ptr<ast::VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}
void ASTStencilBody::visit(const std::shared_ptr<ast::StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}
void ASTStencilBody::visit(const std::shared_ptr<ast::BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<ast::StencilFunCallExpr>& expr) {
  DAWN_ASSERT_MSG(0, "StencilFunCallExpr not allowed in this context");
}
void ASTStencilBody::visit(const std::shared_ptr<ast::StencilFunArgExpr>& expr) {
  DAWN_ASSERT_MSG(0, "StencilFunArgExpr not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<ast::ReturnStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "Return not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<ast::VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int AccessID = iir::getAccessID(expr);

  if(metadata_.isAccessType(iir::FieldAccessType::GlobalVariable, AccessID)) {
    ss_ << "globals." << name;
  } else {
    ss_ << name;

    if(expr->isArrayAccess()) {
      ss_ << "[";
      expr->getIndex()->accept(*this);
      ss_ << "]";
    }
  }
}

void ASTStencilBody::visit(const std::shared_ptr<ast::AssignmentExpr>& expr) {
  expr->getLeft()->accept(*this);
  ss_ << " " << expr->getOp() << " ";
  expr->getRight()->accept(*this);
}

std::string ASTStencilBody::makeIndexString(const std::shared_ptr<ast::FieldAccessExpr>& expr,
                                            std::string kiterStr) {
  bool isVertical = metadata_.getFieldDimensions(iir::getAccessID(expr)).isVertical();
  if(isVertical) {
    return kiterStr;
  }

  bool isHorizontal = !metadata_.getFieldDimensions(iir::getAccessID(expr)).K();
  bool isFullField = !isHorizontal && !isVertical;
  auto unstrDims = ast::dimension_cast<const ast::UnstructuredFieldDimension&>(
      metadata_.getFieldDimensions(iir::getAccessID(expr)).getHorizontalFieldDimension());
  bool isDense = unstrDims.isDense();
  bool isSparse = unstrDims.isSparse();

  std::string denseSize = locToStrideString(unstrDims.getDenseLocationType());

  std::string nbhIdx = "nbhIdx";
  std::string nbhIter = "nbhIdx";

  if(offsets_.has_value()) {
    nbhIdx = std::to_string(offsets_->front());
    nbhIter = std::to_string(offsets_->front());
    offsets_->pop_front();
  }

  if(isFullField && isDense) {
    if((parentIsReduction_ || parentIsForLoop_) &&
       ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
           .hasOffset()) {
      return kiterStr + "*" + denseSize + " + " + nbhIdx;
    } else {
      return kiterStr + "*" + denseSize + "+ pidx";
    }
  }

  if(isFullField && isSparse) {
    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");
    return "nbhIter * kSize * " + denseSize + " + " + kiterStr + "*" + denseSize + " + pidx";
  }

  if(isHorizontal && isDense) {
    if((parentIsReduction_ || parentIsForLoop_) &&
       ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
           .hasOffset()) {
      return nbhIdx;
    } else {
      return "pidx";
    }
  }

  if(isHorizontal && isSparse) {
    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");
    std::string sparseSize = chainToSparseSizeString(unstrDims.getIterSpace());
    return nbhIter + " * " + denseSize + " + pidx";
  }

  DAWN_ASSERT_MSG(false, "Bad Field configuration found in code gen!");
  return "BAD_FIELD_CONFIG";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) {
  if(!expr->getOffset().hasVerticalIndirection()) {
    ss_ << expr->getName() + "[" +
               makeIndexString(expr, "(kIter + " +
                                         std::to_string(expr->getOffset().verticalShift()) + ")") +
               "]";
  } else {
    auto vertOffset = makeIndexString(std::static_pointer_cast<ast::FieldAccessExpr>(
                                          expr->getOffset().getVerticalIndirectionFieldAsExpr()),
                                      "kIter");
    ss_ << expr->getName() + "[" +
               makeIndexString(expr, "(int)(" +
                                         expr->getOffset().getVerticalIndirectionFieldName() + "[" +
                                         vertOffset + "] " + " + " +
                                         std::to_string(expr->getOffset().verticalShift()) + ")") +
               "]";
  }
}

void ASTStencilBody::visit(const std::shared_ptr<ast::FunCallExpr>& expr) {
  std::string callee = expr->getCallee();
  // TODO: temporary hack to remove namespace prefixes
  std::size_t lastcolon = callee.find_last_of(":");
  ss_ << callee.substr(lastcolon + 1) << "(";

  std::size_t numArgs = expr->getArguments().size();
  for(std::size_t i = 0; i < numArgs; ++i) {
    expr->getArguments()[i]->accept(*this);
    ss_ << (i == numArgs - 1 ? "" : ", ");
  }
  ss_ << ")";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::IfStmt>& stmt) {
  ss_ << "if(";
  stmt->getCondExpr()->accept(*this);
  ss_ << ")\n";
  ss_ << "{";
  stmt->getThenStmt()->accept(*this);
  ss_ << "}";
  if(stmt->hasElse()) {
    ss_ << std::string(indent_, ' ') << "else\n";
    ss_ << "{";
    stmt->getElseStmt()->accept(*this);
    ss_ << "}";
  }
}

void ASTStencilBody::visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) {
  DAWN_ASSERT_MSG(!parentIsReduction_,
                  "Nested Reductions not yet supported for CUDA code generation");

  std::string lhs_name = "lhs_" + std::to_string(expr->getID());

  if(!firstPass_) {
    ss_ << " " << lhs_name << " ";
    return;
  }

  parentIsReduction_ = true;

  std::string weights_name = "weights_" + std::to_string(expr->getID());
  ss_ << "::dawn::float_type " << lhs_name << " = ";
  expr->getInit()->accept(*this);
  ss_ << ";\n";
  auto weights = expr->getWeights();

  if(!expr->getOffsets().empty()) {
    offsets_ = std::deque<int>(expr->getOffsets().begin(), expr->getOffsets().end());
  }
  if(weights.has_value()) {
    ss_ << "::dawn::float_type " << weights_name << "[" << weights->size() << "] = {";
    bool first = true;
    for(auto weight : *weights) {
      if(!first) {
        ss_ << ", ";
      }
      weight->accept(*this);
      first = false;
    }
    ss_ << "};\n";
  }
  offsets_ = std::nullopt;
  ss_ << "for (int nbhIter = 0; nbhIter < " << chainToSparseSizeString(expr->getIterSpace())
      << "; nbhIter++)";

  ss_ << "{\n";
  ss_ << "int nbhIdx = " << chainToTableString(expr->getIterSpace()) << "["
      << "pidx * " << chainToSparseSizeString(expr->getIterSpace()) << " + nbhIter"
      << "];\n";
  ss_ << "if (nbhIdx == DEVICE_MISSING_VALUE) { continue; }";
  if(!expr->isArithmetic()) {
    ss_ << lhs_name << " = " << expr->getOp() << "(" << lhs_name << ", ";
  } else {
    ss_ << lhs_name << " " << expr->getOp() << "= ";
  }
  if(weights.has_value()) {
    ss_ << weights_name << "[nbhIter] * ";
  }
  expr->getRhs()->accept(*this);
  if(!expr->isArithmetic()) {
    ss_ << ")";
  }
  ss_ << ";}\n";
  parentIsReduction_ = false;
}

std::string ASTStencilBody::getName(const std::shared_ptr<ast::VarDeclStmt>& stmt) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(stmt));
}
std::string ASTStencilBody::getName(const std::shared_ptr<ast::Expr>& expr) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(expr));
}

ASTStencilBody::ASTStencilBody(const iir::StencilMetaInformation& metadata, const Padding& padding)
    : metadata_(metadata), padding_(padding) {}
ASTStencilBody::~ASTStencilBody() {}

} // namespace cudaico
} // namespace codegen
} // namespace dawn
