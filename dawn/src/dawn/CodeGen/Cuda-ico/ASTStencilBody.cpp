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

  if(isFullField && isDense) {
    if((parentIsReduction_ || parentIsForLoop_) &&
       ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
           .hasOffset()) {
      return kiterStr + "*" + denseSize + "+ nbhIdx";
    } else {
      return kiterStr + "*" + denseSize + "+ " + pidx();
    }
  }

  if(isFullField && isSparse) {
    std::cout << "Parent " << parentIsForLoop_ << "," << parentIsReduction_ << std::endl;
    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");
    return "nbhIter * kSize * " + denseSize + " + " + kiterStr + "*" + denseSize + " + " + pidx();
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
    std::cout << "Parent " << parentIsForLoop_ << "," << parentIsReduction_ << std::endl;

    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");
    std::string sparseSize = chainToSparseSizeString(unstrDims.getIterSpace());
    return "nbhIter * " + denseSize + " + " + pidx();
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

void ASTStencilBody::generateLambda(std::stringstream& ss) const {

  for(auto& it : reductionParser_) {
    const std::unique_ptr<ASTStencilBody>& tt = it.second;
    // move instead?
    tt->generateLambda(ss);
    ss << (*tt).ss_.rdbuf();
  }
}

void ASTStencilBody::visit(const std::shared_ptr<ast::ExprStmt>& stmt) {
  FindReduceOverNeighborExpr findReduceOverNeighborExpr;
  stmt->getExpr()->accept(findReduceOverNeighborExpr);

  // code generate lambda reductions on a second pass
  if(findReduceOverNeighborExpr.hasReduceOverNeighborExpr()) {
    ASTStencilBody astParser(metadata_, padding_);
    stmt->getExpr()->accept(astParser);

    astParser.generateLambda(ss_);

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
    ASTStencilBody astParser(metadata_, padding_);
    stmt->accept(astParser);

    astParser.generateLambda(ss_);

    reductionParser_ = std::move(astParser.reductionParser_);
  }

  ASTCodeGenCXX::visit(stmt);
}

bool ASTStencilBody::hasIrregularPentagons(const std::vector<ast::LocationType>& chain) {
  DAWN_ASSERT(chain.size() > 1);
  return (std::count(chain.begin(), chain.end() - 1, ast::LocationType::Vertices) != 0);
}

std::string ASTStencilBody::nbhLambdaName(const std::shared_ptr<ast::Expr>& expr) {
  return "nbh_lambda" + std::to_string(expr->getID());
}

std::string ASTStencilBody::pidx() {
  if(parentIsReduction_)
    return "fpidx";
  else
    return "pidx";
}

void ASTStencilBody::evalNeighbourReductionLambda(
    const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) {

  ss_ << "auto " << nbhLambdaName(expr) << " = [&] (int fpidx) {" << std::endl;
  auto lhs_name = "lhs_" + std::to_string(expr->getID());

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

  ss_ << "for (int nbhIter = 0; nbhIter < " << chainToSparseSizeString(expr->getIterSpace())
      << "; nbhIter++)";

  ss_ << "{\n";
  ss_ << "int nbhIdx = " << chainToTableString(expr->getIterSpace()) << "["
      << "fpidx * " << chainToSparseSizeString(expr->getIterSpace()) << " + nbhIter"
      << "];\n";

  if(hasIrregularPentagons(expr->getNbhChain())) {
    ss_ << "if (nbhIdx == DEVICE_MISSING_VALUE) { continue; }";
  }

  if(!expr->isArithmetic()) {
    ss_ << lhs_name << " = " << expr->getOp() << "(" << lhs_name << ", ";
  } else {
    ss_ << lhs_name << " " << expr->getOp() << "= ";
  }
  if(expr->getWeights().has_value()) {
    std::string weights_name = "weights_" + std::to_string(expr->getID());
    ss_ << weights_name << "[nbhIter] * ";
  }
  expr->getRhs()->accept(*this);

  if(!expr->isArithmetic()) {
    ss_ << ");\n";
  } else {
    ss_ << ";\n";
  }
  ss_ << "}\n";

  ss_ << "return " << lhs_name << ";};\n";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) {

  std::string lhs_name = nbhLambdaName(expr) + "(" + pidx() + ")";

  parentIsReduction_ = true;

  reductionParser_.emplace(expr->getID(), std::make_unique<ASTStencilBody>(metadata_, padding_));
  reductionParser_.at(expr->getID())->parentIsReduction_ = true;
  reductionParser_.at(expr->getID())->evalNeighbourReductionLambda(expr);
  ss_ << lhs_name;
  return;
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
