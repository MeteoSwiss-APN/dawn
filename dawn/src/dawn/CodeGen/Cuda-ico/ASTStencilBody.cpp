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
  for(const auto& s : stmt->getStatements()) {
    ss_ << indent;
    s->accept(*this);
  }
  indent_ -= DAWN_PRINT_INDENT;
}

std::string ASTStencilBody::nbhIterStr() { return "nbhIter" + std::to_string(recursiveIterNest_); }

std::string ASTStencilBody::nbhIdx_m1Str() {
  return "nbhIdx" + std::to_string(recursiveIterNest_ - 1);
}

std::string ASTStencilBody::nbhIdxStr() { return "nbhIdx" + std::to_string(recursiveIterNest_); }

void ASTStencilBody::visit(const std::shared_ptr<ast::LoopStmt>& stmt) {
  const auto maybeChainPtr =
      dynamic_cast<const ast::ChainIterationDescr*>(stmt->getIterationDescrPtr());
  DAWN_ASSERT_MSG(maybeChainPtr, "general loop concept not implemented yet!\n");

  parentIsForLoop_ = true;
  ss_ << "for (int " + nbhIterStr() + " = 0; " + nbhIterStr() + " < "
      << chainToSparseSizeString(maybeChainPtr->getIterSpace()) << "; " + nbhIterStr() + "++)";

  ss_ << "{\n";
  ss_ << "int " + nbhIdxStr() + " = " << chainToTableString(maybeChainPtr->getIterSpace()) << "["
      << "pidx * " << chainToSparseSizeString(maybeChainPtr->getIterSpace()) << " + " + nbhIterStr()
      << "];\n";
  // if(hasIrregularPentagons(maybeChainPtr->getChain())) {
  if(true) {
    ss_ << "if (" + nbhIdxStr() + " == DEVICE_MISSING_VALUE) { continue; }";
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

void ASTStencilBody::visit(const std::shared_ptr<ast::LiteralAccessExpr>& expr) {
  std::string type(ASTCodeGenCXX::builtinTypeIDToCXXType(expr->getBuiltinType(), false));
  ss_ << (type.empty() ? "" : "(" + type + ") ") << expr->getValue();
}

void ASTStencilBody::visit(const std::shared_ptr<ast::UnaryOperator>& expr) {
  ss_ << "(" << expr->getOp();
  expr->getOperand()->accept(*this);
  ss_ << ")";
}
void ASTStencilBody::visit(const std::shared_ptr<ast::BinaryOperator>& expr) {
  ss_ << "(";
  expr->getLeft()->accept(*this);
  ss_ << " " << expr->getOp() << " ";
  expr->getRight()->accept(*this);
  ss_ << ")";
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

  // 3D
  if(isFullField && isDense) {
    if((parentIsReduction_ || parentIsForLoop_) &&
       ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
           .hasOffset()) {
      // if the field access has an horizontal offset we use the neighbour reduction index
      return kiterStr + "*" + denseSize + "+ " + nbhIdxStr();
    } else {
      // otherwise the main pidx grid point index
      return kiterStr + "*" + denseSize + "+ " + pidx();
    }
  }

  if(isFullField && isSparse) {
    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");
    return nbhIterStr() + " * kSize * " + denseSize + " + " + kiterStr + "*" + denseSize + " + " +
           pidx();
  }

  // 2D
  if(isHorizontal && isDense) {
    if((parentIsReduction_ || parentIsForLoop_) &&
       ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
           .hasOffset()) {
      return nbhIdxStr();
    } else {
      return "pidx";
    }
  }

  if(isHorizontal && isSparse) {
    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");
    std::string sparseSize = chainToSparseSizeString(unstrDims.getIterSpace());
    return nbhIterStr() + " * " + denseSize + " + " + pidx();
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

void ASTStencilBody::generateNeighbourRedLoop(std::stringstream& ss) const {

  for(auto& it : reductionParser_) {
    const std::unique_ptr<ASTStencilBody>& tt = it.second;
    ss << (*tt).ss_.rdbuf();
  }
}

void ASTStencilBody::visit(const std::shared_ptr<ast::ExprStmt>& stmt) {
  FindReduceOverNeighborExpr findReduceOverNeighborExpr;
  stmt->getExpr()->accept(findReduceOverNeighborExpr);

  // if there are neighbour reductions, we preprocess them in advance in order to be able to code
  // generate the loop over neighbour reduction before the expression stmt
  if(findReduceOverNeighborExpr.hasReduceOverNeighborExpr()) {
    // instantiate a new ast stencil body to parse exclusively the neighbour reductions
    ASTStencilBody astParser(metadata_, recursiveIterNest_);
    stmt->getExpr()->accept(astParser);

    // code generate the loop over neighbours reduction
    astParser.generateNeighbourRedLoop(ss_);
    reductionParser_ = std::move(astParser.reductionParser_);
  }

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::VarDeclStmt>& stmt) {
  FindReduceOverNeighborExpr findReduceOverNeighborExpr;
  stmt->accept(findReduceOverNeighborExpr);

  // code generate lambda reductions on a second pass
  if(findReduceOverNeighborExpr.hasReduceOverNeighborExpr()) {
    ASTStencilBody astParser(metadata_, recursiveIterNest_);
    stmt->accept(astParser);

    astParser.generateNeighbourRedLoop(ss_);

    reductionParser_ = std::move(astParser.reductionParser_);
  }

  ASTCodeGenCXX::visit(stmt);
}

bool ASTStencilBody::hasIrregularPentagons(const std::vector<ast::LocationType>& chain) {
  DAWN_ASSERT(chain.size() > 1);
  return (std::count(chain.begin(), chain.end() - 1, ast::LocationType::Vertices) != 0);
}

std::string ASTStencilBody::nbhLhsName(const std::shared_ptr<ast::Expr>& expr) {
  return "lhs_" + std::to_string(expr->getID());
}

std::string ASTStencilBody::pidx() {
  // the pidx within a nested neighbour reduction is the parent neighbor reduction loop index
  if(recursiveIterNest_ > 0)
    return nbhIdx_m1Str();
  else
    return "pidx";
}

void ASTStencilBody::evalNeighbourReductionLambda(
    const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) {

  auto lhs_name = nbhLhsName(expr);

  std::string weights_name = "weights_" + std::to_string(expr->getID());
  ss_ << "::dawn::float_type " << lhs_name << " = ";
  expr->getInit()->accept(*this);
  ss_ << ";\n";
  auto weights = expr->getWeights();
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

  ss_ << "for (int " + nbhIterStr() + " = 0; " + nbhIterStr() + " < "
      << chainToSparseSizeString(expr->getIterSpace()) << "; " + nbhIterStr() + "++)";

  ss_ << "{\n";
  ss_ << "int " + nbhIdxStr() + " = " << chainToTableString(expr->getIterSpace()) << "[" << pidx()
      << " * " << chainToSparseSizeString(expr->getIterSpace()) << " + " + nbhIterStr() << "];\n";

  // TODO remove this hack
  // if(hasIrregularPentagons(expr->getNbhChain())) {
  if(true) {
    ss_ << "if (" + nbhIdxStr() + " == DEVICE_MISSING_VALUE) { continue; }";
  }

  FindReduceOverNeighborExpr findReduceOverNeighborExpr;
  expr->getRhs()->accept(findReduceOverNeighborExpr);

  // here we have finished generating the loop over neighbours structure
  // before we generate the expression, we check if (and generate) nested neighbour reductions
  if(findReduceOverNeighborExpr.hasReduceOverNeighborExpr()) {
    ASTStencilBody astParser(metadata_, recursiveIterNest_ + 1);
    expr->getRhs()->accept(astParser);

    for(auto& redParser : astParser.reductionParser_) {
      ss_ << (redParser.second)->ss_.str();
    }
  }

  if(!expr->isArithmetic()) {
    ss_ << lhs_name << " = " << expr->getOp() << "(" << lhs_name << ", ";
  } else {
    ss_ << lhs_name << " " << expr->getOp() << "= ";
  }
  if(expr->getWeights().has_value()) {
    std::string weights_name = "weights_" + std::to_string(expr->getID());
    ss_ << weights_name << "[" + nbhIterStr() + "] * ";
  }

  expr->getRhs()->accept(*this);

  if(!expr->isArithmetic()) {
    ss_ << ");\n";
  } else {
    ss_ << ";\n";
  }
  ss_ << "}\n";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) {

  std::string lhs_name = nbhLhsName(expr);

  reductionParser_.emplace(expr->getID(),
                           std::make_unique<ASTStencilBody>(metadata_, recursiveIterNest_));
  reductionParser_.at(expr->getID())->parentIsReduction_ = true;
  reductionParser_.at(expr->getID())->evalNeighbourReductionLambda(expr);
  ss_ << lhs_name;
}

std::string ASTStencilBody::getName(const std::shared_ptr<ast::VarDeclStmt>& stmt) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(stmt));
}
std::string ASTStencilBody::getName(const std::shared_ptr<ast::Expr>& expr) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(expr));
}

ASTStencilBody::ASTStencilBody(const iir::StencilMetaInformation& metadata, int recursiveIterNest)
    : metadata_(metadata), recursiveIterNest_(recursiveIterNest) {}
ASTStencilBody::~ASTStencilBody() {}

} // namespace cudaico
} // namespace codegen
} // namespace dawn
