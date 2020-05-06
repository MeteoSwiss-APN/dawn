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
#include <sstream>

namespace dawn {
namespace codegen {
namespace cudaico {

namespace {
class FindReduceOverNeighborExpr : public dawn::ast::ASTVisitorForwarding {
  std::optional<std::shared_ptr<dawn::iir::ReductionOverNeighborExpr>> foundReduction_ =
      std::nullopt;

public:
  void visit(const std::shared_ptr<dawn::iir::ReductionOverNeighborExpr>& stmt) override {
    foundReduction_ = stmt;
    return;
  }
  bool hasReduceOverNeighborExpr() const { return foundReduction_.has_value(); }
};
} // namespace

void ASTStencilBody::visit(const std::shared_ptr<iir::BlockStmt>& stmt) {
  indent_ += DAWN_PRINT_INDENT;
  auto indent = std::string(indent_, ' ');
  for(const auto& s : stmt->getStatements()) {
    ss_ << indent;
    s->accept(*this);
  }
  indent_ -= DAWN_PRINT_INDENT;
}
void ASTStencilBody::visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}
void ASTStencilBody::visit(const std::shared_ptr<iir::LoopStmt>& stmt) {
  const auto maybeChainPtr =
      dynamic_cast<const ast::ChainIterationDescr*>(stmt->getIterationDescrPtr());
  DAWN_ASSERT_MSG(maybeChainPtr, "general loop concept not implemented yet!\n");

  ss_ << "for (int nbhIter = 0; nbhIter < " << chainToSparseSizeString(maybeChainPtr->getChain())
      << "; nbhIter++)";
  parentIsForLoop_ = true;

  ss_ << "{\n";
  ss_ << "int nbhIdx = __ldg(&" << chainToTableString(maybeChainPtr->getChain()) << "["
      << "pidx * " << chainToSparseSizeString(maybeChainPtr->getChain()) << " + nbhIter"
      << "]);\n";
  ss_ << "if (nbhIdx == DEVICE_MISSING_VALUE) { continue; }";

  stmt->getBlockStmt()->accept(*this);

  ss_ << "}\n";
  parentIsForLoop_ = false;
}
void ASTStencilBody::visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}
void ASTStencilBody::visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}
void ASTStencilBody::visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
  DAWN_ASSERT_MSG(0, "StencilFunCallExpr not allowed in this context");
}
void ASTStencilBody::visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {
  DAWN_ASSERT_MSG(0, "StencilFunArgExpr not allowed in this context");
}
void ASTStencilBody::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {}

void ASTStencilBody::visit(const std::shared_ptr<iir::AssignmentExpr>& expr) {
  FindReduceOverNeighborExpr reductionFinder;
  expr->getRight()->accept(reductionFinder);
  if(reductionFinder.hasReduceOverNeighborExpr()) {
    expr->getRight()->accept(*this);
    expr->getLeft()->accept(*this);
    ss_ << expr->getOp() << "lhs";
  } else {
    expr->getLeft()->accept(*this);
    ss_ << " " << expr->getOp() << " ";
    expr->getRight()->accept(*this);
  }
}

void ASTStencilBody::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  auto unstrDims = sir::dimension_cast<const sir::UnstructuredFieldDimension&>(
      metadata_.getFieldDimensions(iir::getAccessID(expr)).getHorizontalFieldDimension());
  std::string denseOffset =
      "kIter * " + locToDenseSizeStringGpuMesh(unstrDims.getDenseLocationType());
  if(unstrDims.isDense()) {
    std::string resArgName;
    if(!parentIsReduction_ && parentIsForLoop_ &&
       ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
           .hasOffset()) {
      resArgName = denseOffset + " + nbhIdx";
    } else {
      resArgName = denseOffset + " + pidx";
    }
    ss_ << getName(expr) << "[" << resArgName << "]";
  } else {
    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");

    std::string sparseSize = chainToSparseSizeString(unstrDims.getNeighborChain());
    std::string resArgName = denseOffset + " * " + sparseSize + "+ nbhIter * " +
                             locToDenseSizeStringGpuMesh(unstrDims.getDenseLocationType()) +
                             "+ pidx";
    ss_ << getName(expr) << "[" << resArgName << "]";
  }
}

void ASTStencilBody::visit(const std::shared_ptr<iir::FunCallExpr>& expr) {
  std::string callee = expr->getCallee();
  // temporary hack to remove the "math::" prefix
  ss_ << callee.substr(6, callee.size()) << "(";

  std::size_t numArgs = expr->getArguments().size();
  for(std::size_t i = 0; i < numArgs; ++i) {
    expr->getArguments()[i]->accept(*this);
    ss_ << (i == numArgs - 1 ? "" : ", ");
  }
  ss_ << ")";
}

void ASTStencilBody::visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) {
  ss_ << "::dawn::float_type lhs = ";
  expr->getInit()->accept(*this);
  ss_ << ";\n";
  auto weights = expr->getWeights();
  if(weights.has_value()) {
    ss_ << "::dawn::float_type weights[" << weights->size() << "] = {";
    bool first = true;
    for(auto weight : *weights) {
      if(!first) {
        ss_ << ", ";
      }
      weight->accept(*this);
      first = false;
    }
    ss_ << "};";
  }
  ss_ << "for (int nbhIter = 0; nbhIter < " << chainToSparseSizeString(expr->getNbhChain())
      << "; nbhIter++)";

  parentIsReduction_ = true;
  ss_ << "{\n";
  ss_ << "int nbhIdx = __ldg(&" << chainToTableString(expr->getNbhChain()) << "["
      << "pidx * " << chainToSparseSizeString(expr->getNbhChain()) << " + nbhIter"
      << "]);\n";
  ss_ << "if (nbhIdx == DEVICE_MISSING_VALUE) { continue; }";
  ss_ << "lhs " << expr->getOp() << "=";
  if(weights.has_value()) {
    ss_ << " weights[nbhIter] * ";
  }
  expr->getRhs()->accept(*this);
  ss_ << ";}\n";
  parentIsReduction_ = false;
}

std::string ASTStencilBody::getName(const std::shared_ptr<iir::VarDeclStmt>& stmt) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(stmt));
}
std::string ASTStencilBody::getName(const std::shared_ptr<iir::Expr>& expr) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(expr));
}

ASTStencilBody::ASTStencilBody(const iir::StencilMetaInformation& metadata) : metadata_(metadata) {}
ASTStencilBody::~ASTStencilBody() {}

} // namespace cudaico
} // namespace codegen
} // namespace dawn