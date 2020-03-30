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

#include "WeightsChecker.h"
#include "dawn/AST/ASTFwd.h"
#include "dawn/Support/SourceLocation.h"

namespace dawn {

void WeightChecker::WeightCheckerImpl::visit(
    const std::shared_ptr<dawn::ast::FieldAccessExpr>& fieldAccessExpr) {

  auto fieldName = fieldAccessExpr->getName();
  // the name in the FieldAccessExpr may be stale if the there are nested stencils
  // in this case we need to look up the new AccessID in the data of the fieldAccessExpr
  if(fieldAccessExpr->hasData()) {
    auto newAccessID = fieldAccessExpr->getData<iir::IIRAccessExprData>().AccessID;
    if(newAccessID.has_value()) {
      DAWN_ASSERT(idToNameMap_.count(newAccessID.value()));
      fieldName = idToNameMap_.at(newAccessID.value());
    }
  }

  DAWN_ASSERT(nameToDimensions_.count(fieldName));
  weightsValid_ = sir::dimension_cast<const sir::UnstructuredFieldDimension&>(
                      nameToDimensions_.at(fieldName).getHorizontalFieldDimension())
                      .isDense();
}
void WeightChecker::WeightCheckerImpl::visit(const std::shared_ptr<dawn::ast::FunCallExpr>& expr) {
  weightsValid_ = false;
}
void WeightChecker::WeightCheckerImpl::visit(
    const std::shared_ptr<dawn::ast::StencilFunCallExpr>& expr) {
  weightsValid_ = false;
}
void WeightChecker::WeightCheckerImpl::visit(
    const std::shared_ptr<dawn::ast::StencilFunArgExpr>& expr) {
  weightsValid_ = false;
}
void WeightChecker::WeightCheckerImpl::visit(
    const std::shared_ptr<dawn::ast::ReductionOverNeighborExpr>& expr) {
  weightsValid_ = false; // TODO nested reduce over neighbors
}

bool WeightChecker::WeightCheckerImpl::isValid() const { return weightsValid_; }

WeightChecker::WeightCheckerImpl::WeightCheckerImpl(
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap)
    : nameToDimensions_(nameToDimensionsMap) {}

WeightChecker::WeightCheckerImpl::WeightCheckerImpl(
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap,
    const std::unordered_map<int, std::string> idToNameMap)
    : nameToDimensions_(nameToDimensionsMap), idToNameMap_(idToNameMap) {}

WeightChecker::ConsistencyResult
WeightChecker::CheckWeights(const iir::IIR& iir, const iir::StencilMetaInformation& metaData) {

  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(iir)) {
    for(const auto& stmt : doMethod->getAST().getStatements()) {
      if(stmt->getKind() == ast::Stmt::Kind::ExprStmt) {
        auto exprPtr = static_cast<ast::ExprStmt*>(stmt.get())->getExpr();
        if(exprPtr->getKind() == ast::Expr::Kind::ReductionOverNeighborExpr) {
          auto reduceExprPtr = static_cast<ast::ReductionOverNeighborExpr*>(exprPtr.get());
          auto weights = reduceExprPtr->getWeights();
          if(!weights.has_value()) {
            continue;
          }
          for(const auto& weightPtr : *weights) {
            WeightChecker::WeightCheckerImpl checker(doMethod->getFieldDimensionsByName(),
                                                     metaData.getAccessIDToNameMap());
            weightPtr->accept(checker);
            if(!checker.isValid()) {
              return {false, weightPtr->getSourceLocation()};
            }
          }
        }
      }
    }
  }
  return {true, SourceLocation()};
}

WeightChecker::ConsistencyResult WeightChecker::CheckWeights(const SIR& sir) {
  for(const auto& stencil : sir.Stencils) {
    DAWN_ASSERT(stencil);
    std::unordered_map<std::string, sir::FieldDimensions> stencilFieldDims;
    for(const auto& field : stencil->Fields) {
      stencilFieldDims.insert({field->Name, field->Dimensions});
    }
    const auto& stencilAst = stencil->StencilDescAst;
    for(const auto& stmt : stencilAst->getRoot()->getChildren()) {
      if(stmt->getKind() == ast::Stmt::Kind::ExprStmt) {
        auto exprPtr = static_cast<ast::ExprStmt*>(stmt.get())->getExpr();
        if(exprPtr->getKind() == ast::Expr::Kind::ReductionOverNeighborExpr) {
          auto reduceExprPtr = static_cast<ast::ReductionOverNeighborExpr*>(exprPtr.get());
          auto weights = reduceExprPtr->getWeights();
          if(!weights.has_value()) {
            continue;
          }
          for(const auto& weightPtr : *weights) {
            WeightChecker::WeightCheckerImpl checker(stencilFieldDims);
            weightPtr->accept(checker);
            if(!checker.isValid()) {
              return {false, weightPtr->getSourceLocation()};
            }
          }
        }
      }
    }
  }

  return {true, SourceLocation()};
}

} // namespace dawn