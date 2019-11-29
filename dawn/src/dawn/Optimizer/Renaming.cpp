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
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/MultiStage.h"
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
    int& varAccessID = *stmt->getData<iir::VarDeclStmtData>().AccessID;
    if(varAccessID == oldAccessID_)
      varAccessID = newAccessID_;
    iir::ASTVisitorForwarding::visit(stmt);
    stmt->getName() = instantiation_->getNameFromAccessID(varAccessID);
  }

  virtual void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override {
    std::shared_ptr<iir::StencilFunctionInstantiation> fun =
        instantiation_->getStencilFunctionInstantiation(expr);
    renameCallerAccessIDInStencilFunction(fun.get(), oldAccessID_, newAccessID_);
    iir::ASTVisitorForwarding::visit(expr);
  }

  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    int& varAccessID = *expr->getData<iir::IIRAccessExprData>().AccessID;
    if(varAccessID == oldAccessID_)
      varAccessID = newAccessID_;
    expr->setName(instantiation_->getNameFromAccessID(varAccessID));
  }

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    int& fieldAccessID = *expr->getData<iir::IIRAccessExprData>().AccessID;
    if(fieldAccessID == oldAccessID_) {
      fieldAccessID = newAccessID_;
      expr->setName(instantiation_->getNameFromAccessID(fieldAccessID));
    }
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

void renameAccessIDInStmts(iir::StencilMetaInformation* metadata, int oldAccessID, int newAccessID,
                           ArrayRef<std::shared_ptr<iir::Stmt>> stmts) {
  AccessIDRemapper<iir::StencilMetaInformation> remapper(metadata, oldAccessID, newAccessID);

  for(auto& stmt : stmts)
    stmt->accept(remapper);
}

void renameAccessIDInStmts(iir::StencilFunctionInstantiation* instantiation, int oldAccessID,
                           int newAccessID, ArrayRef<std::shared_ptr<iir::Stmt>> stmts) {
  AccessIDRemapper<iir::StencilFunctionInstantiation> remapper(instantiation, oldAccessID,
                                                               newAccessID);

  for(const auto& stmt : stmts)
    stmt->accept(remapper);
}

void renameAccessIDInExpr(iir::StencilInstantiation* instantiation, int oldAccessID,
                          int newAccessID, std::shared_ptr<iir::Expr>& expr) {
  AccessIDRemapper<iir::StencilMetaInformation> remapper(&(instantiation->getMetaData()),
                                                         oldAccessID, newAccessID);
  expr->accept(remapper);
}

void renameAccessIDInAccesses(const iir::StencilMetaInformation* metadata, int oldAccessID,
                              int newAccessID, ArrayRef<std::shared_ptr<iir::Stmt>> stmts) {
  for(auto& stmt : stmts) {
    auto& callerAccesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;
    renameAccessesMaps(callerAccesses->getReadAccesses(), oldAccessID, newAccessID);
    renameAccessesMaps(callerAccesses->getWriteAccesses(), oldAccessID, newAccessID);
  }
}

void renameAccessIDInAccesses(iir::StencilFunctionInstantiation* instantiation, int oldAccessID,
                              int newAccessID, ArrayRef<std::shared_ptr<iir::Stmt>> stmts) {
  for(auto& stmt : stmts) {
    auto& callerAccesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;
    renameAccessesMaps(callerAccesses->getReadAccesses(), oldAccessID, newAccessID);
    renameAccessesMaps(callerAccesses->getWriteAccesses(), oldAccessID, newAccessID);
    auto& calleeAccesses = stmt->getData<iir::IIRStmtData>().CalleeAccesses;
    renameAccessesMaps(calleeAccesses->getReadAccesses(), oldAccessID, newAccessID);
    renameAccessesMaps(calleeAccesses->getWriteAccesses(), oldAccessID, newAccessID);
  }
}

void renameAccessIDInMultiStage(iir::MultiStage* multiStage, int oldAccessID, int newAccessID) {
  for(auto stageIt = multiStage->childrenBegin(), stageEnd = multiStage->childrenEnd();
      stageIt != stageEnd; ++stageIt) {
    iir::Stage& stage = (**stageIt);
    for(const auto& doMethodPtr : stage.getChildren()) {
      iir::DoMethod& doMethod = *doMethodPtr;
      renameAccessIDInStmts(&(multiStage->getMetadata()), oldAccessID, newAccessID,
                            doMethod.getAST().getStatements());
      renameAccessIDInAccesses(&(multiStage->getMetadata()), oldAccessID, newAccessID,
                               doMethod.getAST().getStatements());
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
  renameAccessIDInStmts(function, oldAccessID, newAccessID,
                        function->getDoMethod()->getAST().getStatements());

  // Update accesses
  renameAccessIDInAccesses(function, oldAccessID, newAccessID,
                           function->getDoMethod()->getAST().getStatements());

  // Recompute the fields
  function->update();
}

} // namespace dawn
