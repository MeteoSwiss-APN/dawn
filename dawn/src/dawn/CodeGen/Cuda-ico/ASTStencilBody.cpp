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
#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/LocationType.h"
#include "dawn/CodeGen/Cuda-ico/ReductionMerger.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace dawn {
namespace codegen {
namespace cudaico {

void ASTStencilBody::visit(const std::shared_ptr<ast::BlockStmt>& stmt) {
  indent_ += DAWN_PRINT_INDENT;
  auto indent = std::string(indent_, ' ');
  currentBlock_ = stmt->getID();
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
  if(hasIrregularPentagons(maybeChainPtr->getChain())) {
    ss_ << "if (nbhIdx == DEVICE_MISSING_VALUE) { continue; }";
  }

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
  std::stringstream& localSS = parentIsReduction_ ? reductionMap_[parentReductionID_] : ss_;
  std::string name = getName(expr);
  int AccessID = iir::getAccessID(expr);

  if(metadata_.isAccessType(iir::FieldAccessType::GlobalVariable, AccessID)) {
    localSS << "globals." << name;
  } else {
    localSS << name;

    if(expr->isArrayAccess()) {
      localSS << "[";
      expr->getIndex()->accept(*this);
      localSS << "]";
    }
  }
}

void ASTStencilBody::visit(const std::shared_ptr<ast::LiteralAccessExpr>& expr) {
  std::stringstream& localSS = parentIsReduction_ ? reductionMap_[parentReductionID_] : ss_;
  std::string type(ASTCodeGenCXX::builtinTypeIDToCXXType(expr->getBuiltinType(), false));
  localSS << (type.empty() ? "" : "(" + type + ") ") << expr->getValue();
}

void ASTStencilBody::visit(const std::shared_ptr<ast::UnaryOperator>& expr) {
  std::stringstream& localSS = parentIsReduction_ ? reductionMap_[parentReductionID_] : ss_;
  localSS << "(" << expr->getOp();
  expr->getOperand()->accept(*this);
  localSS << ")";
}
void ASTStencilBody::visit(const std::shared_ptr<ast::BinaryOperator>& expr) {
  std::stringstream& localSS = parentIsReduction_ ? reductionMap_[parentReductionID_] : ss_;
  localSS << "(";
  expr->getLeft()->accept(*this);
  localSS << " " << expr->getOp() << " ";
  expr->getRight()->accept(*this);
  localSS << ")";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::AssignmentExpr>& expr) {
  std::stringstream& localSS = parentIsReduction_ ? reductionMap_[parentReductionID_] : ss_;
  expr->getLeft()->accept(*this);
  localSS << " " << expr->getOp() << " ";
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

  if(isFullField && isDense) {
    if((parentIsReduction_ || parentIsForLoop_) &&
       ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
           .hasOffset()) {
      return kiterStr + "*" + denseSize + "+ nbhIdx";
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
      return "nbhIdx";
    } else {
      return "pidx";
    }
  }

  if(isHorizontal && isSparse) {
    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");
    std::string sparseSize = chainToSparseSizeString(unstrDims.getIterSpace());
    return "nbhIter * " + denseSize + " + pidx";
  }

  DAWN_ASSERT_MSG(false, "Bad Field configuration found in code gen!");
  return "BAD_FIELD_CONFIG";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) {
  std::stringstream& localSS = parentIsReduction_ ? reductionMap_[parentReductionID_] : ss_;

  if(!expr->getOffset().hasVerticalIndirection()) {
    localSS << expr->getName() + "[" +
                   makeIndexString(expr, "(kIter + " +
                                             std::to_string(expr->getOffset().verticalShift()) +
                                             ")") +
                   "]";
  } else {
    auto vertOffset = makeIndexString(std::static_pointer_cast<ast::FieldAccessExpr>(
                                          expr->getOffset().getVerticalIndirectionFieldAsExpr()),
                                      "kIter");
    localSS << expr->getName() + "[" +
                   makeIndexString(expr,
                                   "(int)(" + expr->getOffset().getVerticalIndirectionFieldName() +
                                       "[" + vertOffset + "] " + " + " +
                                       std::to_string(expr->getOffset().verticalShift()) + ")") +
                   "]";
  }
}

void ASTStencilBody::visit(const std::shared_ptr<ast::FunCallExpr>& expr) {
  std::stringstream& localSS = parentIsReduction_ ? reductionMap_[parentReductionID_] : ss_;

  std::string callee = expr->getCallee();
  // TODO: temporary hack to remove namespace prefixes
  std::size_t lastcolon = callee.find_last_of(":");

  localSS << callee.substr(lastcolon + 1) << "(";

  std::size_t numArgs = expr->getArguments().size();
  for(std::size_t i = 0; i < numArgs; ++i) {
    expr->getArguments()[i]->accept(*this);
    localSS << (i == numArgs - 1 ? "" : ", ");
  }
  localSS << ")";
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

void ASTStencilBody::visit(const std::shared_ptr<ast::ExprStmt>& stmt) {
  FindReduceOverNeighborExpr findReduceOverNeighborExpr;
  stmt->getExpr()->accept(findReduceOverNeighborExpr);

  if(findReduceOverNeighborExpr.hasReduceOverNeighborExpr()) {
    const auto& reductions = findReduceOverNeighborExpr.reduceOverNeighborExprs();
    for(auto red : reductions) {
      if(reductionMap_.count(red->getID())) {
        ss_ << reductionMap_.at(red->getID()).str();
      }
    }
  }

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::VarDeclStmt>& stmt) {
  FindReduceOverNeighborExpr findReduceOverNeighborExpr;
  stmt->accept(findReduceOverNeighborExpr);

  if(findReduceOverNeighborExpr.hasReduceOverNeighborExpr()) {
    const auto& reductions = findReduceOverNeighborExpr.reduceOverNeighborExprs();
    for(auto red : reductions) {
      if(reductionMap_.count(red->getID())) {
        ss_ << reductionMap_.at(red->getID()).str();
      }
    }
  }

  ASTCodeGenCXX::visit(stmt);
}

bool ASTStencilBody::hasIrregularPentagons(const std::vector<ast::LocationType>& chain) {
  DAWN_ASSERT(chain.size() > 1);
  return (std::count(chain.begin(), chain.end() - 1, ast::LocationType::Vertices) != 0);
}

void ASTStencilBody::visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) {
  DAWN_ASSERT_MSG(!parentIsReduction_,
                  "Nested Reductions not yet supported for CUDA code generation");

  std::string lhs_name = "lhs_" + std::to_string(expr->getID());

  if(!firstPass_) {
    ss_ << lhs_name;
    return;
  }

  parentIsReduction_ = true;
  parentReductionID_ = expr->getID();

  std::vector<std::shared_ptr<ast::ReductionOverNeighborExpr>> exprs{expr};
  std::vector<std::string> lhs_names{lhs_name};

  if(blockToMergeGroupMap_.has_value()) {
    // find "our" group
    for(auto group : blockToMergeGroupMap_->at(currentBlock_)) {
      if(std::any_of(group.begin(), group.end(),
                     [expr](const std::shared_ptr<ast::ReductionOverNeighborExpr>& red) {
                       return *red == *expr;
                     })) {
        if(group.size() > 1) {
          // we are in a group
          if(group[0]->getID() != expr->getID()) {
            // but not the first element, lets bail out
            parentIsReduction_ = false;
            parentReductionID_ = -1;
            return;
          }
          exprs = group;
          lhs_names.clear();
          for(const auto& red : group) {
            lhs_names.push_back("lhs_" + std::to_string(red->getID()));
          }
        }
      }
    }
  }

  std::stringstream& localSS_ = reductionMap_[expr->getID()];

  {
    int lhs_iter = 0;
    for(auto expr : exprs) {
      std::string weights_name = "weights_" + std::to_string(expr->getID());
      localSS_ << "::dawn::float_type " << lhs_names[lhs_iter] << " = ";
      lhs_iter++;
      expr->getInit()->accept(*this);
      localSS_ << ";\n";
      auto weights = expr->getWeights();
      if(weights.has_value()) {
        localSS_ << "::dawn::float_type " << weights_name << "[" << weights->size() << "] = {";
        bool first = true;
        for(auto weight : *weights) {
          if(!first) {
            localSS_ << ", ";
          }
          weight->accept(*this);
          first = false;
        }
        localSS_ << "};\n";
      }
    }
  }

  localSS_ << "for (int nbhIter = 0; nbhIter < " << chainToSparseSizeString(expr->getIterSpace())
           << "; nbhIter++)";

  localSS_ << "{\n";
  localSS_ << "int nbhIdx = " << chainToTableString(expr->getIterSpace()) << "["
           << "pidx * " << chainToSparseSizeString(expr->getIterSpace()) << " + nbhIter"
           << "];\n";

  if(hasIrregularPentagons(expr->getNbhChain())) {
    localSS_ << "if (nbhIdx == DEVICE_MISSING_VALUE) { continue; }";
  }

  {
    int lhs_iter = 0;
    for(auto expr : exprs) {
      if(!expr->isArithmetic()) {
        localSS_ << lhs_names[lhs_iter] << " = " << expr->getOp() << "(" << lhs_names[lhs_iter]
                 << ", ";
      } else {
        localSS_ << lhs_names[lhs_iter] << " " << expr->getOp() << "= ";
      }
      lhs_iter++;
      if(expr->getWeights().has_value()) {
        std::string weights_name = "weights_" + std::to_string(expr->getID());
        localSS_ << weights_name << "[nbhIter] * ";
      }
      expr->getRhs()->accept(*this);
      if(!expr->isArithmetic()) {
        localSS_ << ");\n";
      } else {
        localSS_ << ";\n";
      }
    }
  }
  localSS_ << "}\n";
  parentIsReduction_ = false;
  parentReductionID_ = -1;
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
