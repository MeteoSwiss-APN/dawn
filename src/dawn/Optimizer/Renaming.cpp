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

#include "dawn/Optimizer/Renaming.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include <unordered_map>

namespace dawn {

namespace {

/// @brief Remap all accesses from `oldAccessID` to `newAccessID` in all statements
template <class InstantiationType>
class AccessIDRemapper : public iir::ASTVisitorForwarding {
  InstantiationType* instantiation_;

  int oldAccessID_;
  int newAccessID_;

public:
  AccessIDRemapper(InstantiationType* instantiation, int oldAccessID, int newAccessID)
      : instantiation_(instantiation), oldAccessID_(oldAccessID), newAccessID_(newAccessID) {}

  virtual void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {
    int varAccessID = instantiation_->getAccessIDFromStmt(stmt);
    if(varAccessID == oldAccessID_)
      instantiation_->setAccessIDOfStmt(stmt, newAccessID_);
    iir::ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override {
    std::shared_ptr<iir::StencilFunctionInstantiation> fun =
        instantiation_->getStencilFunctionInstantiation(expr);
    renameCallerAccessIDInStencilFunction(fun.get(), oldAccessID_, newAccessID_);
    iir::ASTVisitorForwarding::visit(expr);
  }

  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    int varAccessID = instantiation_->getAccessIDFromExpr(expr);
    if(varAccessID == oldAccessID_)
      instantiation_->setAccessIDOfExpr(expr, newAccessID_);
  }

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    int fieldAccessID = instantiation_->getAccessIDFromExpr(expr);
    if(fieldAccessID == oldAccessID_)
      instantiation_->setAccessIDOfExpr(expr, newAccessID_);
  }
};

/// @brief Remap all accesses from `oldAccessID` to `newAccessID` in the `accessesMap`
static void renameAccessesMaps(std::unordered_map<int, iir::Extents>& accessesMap, int oldAccessID,
                               int newAccessID) {
  for(auto it = accessesMap.begin(); it != accessesMap.end();) {
    if(it->first == oldAccessID) {
      accessesMap.emplace(newAccessID, it->second);
      it = accessesMap.erase(it);
    } else {
      ++it;
    }
  }
}

} // anonymous namespace

void renameAccessIDInStmts(
    iir::StencilMetaInformation* metadata, int oldAccessID, int newAccessID,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) {
  AccessIDRemapper<iir::StencilMetaInformation> remapper(metadata, oldAccessID, newAccessID);

  for(auto& statementAccessesPair : statementAccessesPairs)
    statementAccessesPair->getStatement()->accept(remapper);
}

void renameAccessIDInStmts(
    iir::StencilFunctionInstantiation* instantiation, int oldAccessID, int newAccessID,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) {
  AccessIDRemapper<iir::StencilFunctionInstantiation> remapper(instantiation, oldAccessID,
                                                               newAccessID);

  for(const auto& statementAccessesPair : statementAccessesPairs)
    statementAccessesPair->getStatement()->accept(remapper);
}

void renameAccessIDInExpr(iir::StencilInstantiation* instantiation, int oldAccessID,
                          int newAccessID, std::shared_ptr<iir::Expr>& expr) {
  AccessIDRemapper<iir::StencilMetaInformation> remapper(&(instantiation->getMetaData()),
                                                         oldAccessID, newAccessID);
  expr->accept(remapper);
}

void renameAccessIDInAccesses(
    const iir::StencilMetaInformation* metadata, int oldAccessID, int newAccessID,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) {
  for(auto& statementAccessesPair : statementAccessesPairs) {
    auto& callerAccesses =
        statementAccessesPair->getStatement()->getData<iir::IIRStmtData>().CallerAccesses;
    renameAccessesMaps(callerAccesses->getReadAccesses(), oldAccessID, newAccessID);
    renameAccessesMaps(callerAccesses->getWriteAccesses(), oldAccessID, newAccessID);
  }
}

void renameAccessIDInAccesses(
    iir::StencilFunctionInstantiation* instantiation, int oldAccessID, int newAccessID,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) {
  for(auto& statementAccessesPair : statementAccessesPairs) {
    auto& callerAccesses =
        statementAccessesPair->getStatement()->getData<iir::IIRStmtData>().CallerAccesses;
    renameAccessesMaps(callerAccesses->getReadAccesses(), oldAccessID, newAccessID);
    renameAccessesMaps(callerAccesses->getWriteAccesses(), oldAccessID, newAccessID);
    renameAccessesMaps(callerAccesses->getReadAccesses(), oldAccessID, newAccessID);
    renameAccessesMaps(callerAccesses->getWriteAccesses(), oldAccessID, newAccessID);
  }
}

void renameAccessIDInMultiStage(iir::MultiStage* multiStage, int oldAccessID, int newAccessID) {
  for(auto stageIt = multiStage->childrenBegin(), stageEnd = multiStage->childrenEnd();
      stageIt != stageEnd; ++stageIt) {
    iir::Stage& stage = (**stageIt);
    for(const auto& doMethodPtr : stage.getChildren()) {
      iir::DoMethod& doMethod = *doMethodPtr;
      renameAccessIDInStmts(&(multiStage->getMetadata()), oldAccessID, newAccessID,
                            doMethod.getChildren());
      renameAccessIDInAccesses(&(multiStage->getMetadata()), oldAccessID, newAccessID,
                               doMethod.getChildren());
      doMethod.update(iir::NodeUpdateType::level);
    }
    stage.update(iir::NodeUpdateType::levelAndTreeAbove);
  }
}
void renameAccessIDInStencil(iir::Stencil* stencil, int oldAccessID, int newAccessID) {
  for(const auto& multistage : stencil->getChildren()) {
    renameAccessIDInMultiStage(multistage.get(), oldAccessID, newAccessID);
  }
}

void renameCallerAccessIDInStencilFunction(iir::StencilFunctionInstantiation* function,
                                           int oldAccessID, int newAccessID) {
  // Update argument maps
  for(auto& argumentAccessIDPair : function->ArgumentIndexToCallerAccessIDMap()) {
    int& AccessID = argumentAccessIDPair.second;
    if(AccessID == oldAccessID)
      AccessID = newAccessID;
  }

  function->replaceKeyInMap(function->getCallerAccessIDToInitialOffsetMap(), oldAccessID,
                            newAccessID);

  // // Update AccessID to name map
  function->replaceKeyInMap(function->getAccessIDToNameMap(), oldAccessID, newAccessID);

  // Update statements
  renameAccessIDInStmts(function, oldAccessID, newAccessID, function->getDoMethod()->getChildren());

  // Update accesses
  renameAccessIDInAccesses(function, oldAccessID, newAccessID,
                           function->getDoMethod()->getChildren());

  // Recompute the fields
  function->update();
}

} // namespace dawn
