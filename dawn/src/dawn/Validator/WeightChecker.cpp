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

#include "WeightChecker.h"
#include "dawn/AST/ASTFwd.h"
#include "dawn/Support/SourceLocation.h"

namespace dawn {

std::shared_ptr<const iir::StencilFunctionInstantiation>
WeightChecker::WeightCheckerImpl::getStencilFunctionInstantiation(
    const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
  if(!functionInstantiationStack_.empty())
    return functionInstantiationStack_.top()->getStencilFunctionInstantiation(expr);
  return (*ExprToStencilFunctionInstantiationMap_).get().at(expr);
}

void WeightChecker::WeightCheckerImpl::visit(
    const std::shared_ptr<dawn::ast::FieldAccessExpr>& fieldAccessExpr) {
  if(parentIsWeight_) {
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
}
void WeightChecker::WeightCheckerImpl::visit(const std::shared_ptr<dawn::ast::FunCallExpr>& expr) {
  if(parentIsWeight_) {
    weightsValid_ = false;
    return;
  } else {
    for(const auto& s : expr->getChildren()) {
      s->accept(*this);
    }
  }
}
void WeightChecker::WeightCheckerImpl::visit(
    const std::shared_ptr<dawn::ast::StencilFunCallExpr>& expr) {
  if(parentIsWeight_) {
    weightsValid_ = false;
    return;
  } else {
    // we only need to do this in the case we arrive from IIR, in the SIR the stencil funs are
    // checked separately
    if(ExprToStencilFunctionInstantiationMap_.has_value()) {
      std::shared_ptr<const iir::StencilFunctionInstantiation> funCall =
          getStencilFunctionInstantiation(expr);

      functionInstantiationStack_.push(funCall);

      // Follow the AST of the stencil function, it maybe unused in a nested stencil function
      funCall->getAST()->accept(*this);

      // visit arguments
      for(const auto& s : expr->getChildren()) {
        s->accept(*this);
      }

      functionInstantiationStack_.pop();
    }
  }
}
void WeightChecker::WeightCheckerImpl::visit(
    const std::shared_ptr<dawn::ast::ReductionOverNeighborExpr>& expr) {

  if(expr->getWeights().has_value()) {
    parentIsWeight_ = true;
    for(const auto& weight : *expr->getWeights()) {
      weight->accept(*this);
      if(!weightsValid_) {
        return;
      }
    }
    parentIsWeight_ = false;
  }

  expr->getRhs()->accept(*this);
}

bool WeightChecker::WeightCheckerImpl::isValid() const { return weightsValid_; }

WeightChecker::WeightCheckerImpl::WeightCheckerImpl(
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap)
    : nameToDimensions_(nameToDimensionsMap) {}

WeightChecker::WeightCheckerImpl::WeightCheckerImpl(
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap,
    const std::unordered_map<int, std::string> idToNameMap,
    const std::unordered_map<std::shared_ptr<iir::StencilFunCallExpr>,
                             std::shared_ptr<iir::StencilFunctionInstantiation>>& exprToFunMap)
    : nameToDimensions_(nameToDimensionsMap), idToNameMap_(idToNameMap),
      ExprToStencilFunctionInstantiationMap_(exprToFunMap) {}

WeightChecker::ConsistencyResult
WeightChecker::CheckWeights(const iir::IIR& iir, const iir::StencilMetaInformation& metaData) {
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(iir)) {
    WeightChecker::WeightCheckerImpl checker(doMethod->getFieldDimensionsByName(),
                                             metaData.getAccessIDToNameMap(),
                                             metaData.getExprToStencilFunctionInstantiation());
    for(const auto& stmt : doMethod->getAST().getStatements()) {
      stmt->accept(checker);
      if(!checker.isValid()) {
        return {false, stmt->getSourceLocation()};
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
      WeightChecker::WeightCheckerImpl checker(stencilFieldDims);
      stmt->accept(checker);
      if(!checker.isValid()) {
        return {false, stmt->getSourceLocation()};
      }
    }
  }
  return {true, SourceLocation()};
}

} // namespace dawn