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
#include "dawn/Optimizer/Accesses.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StencilFunctionInstantiation.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/Statement.h"
#include <unordered_map>

namespace dawn {

namespace {

/// @brief Remap all accesses from `oldAccessID` to `newAccessID` in all statements
template <class InstantiationType>
class AccessIDRemapper : public ASTVisitorForwarding {
  InstantiationType* instantiation_;

  int oldAccessID_;
  int newAccessID_;

  std::unordered_map<std::shared_ptr<Expr>, int>& ExprToAccessIDMap_;
  std::unordered_map<std::shared_ptr<Stmt>, int>& StmtToAccessIDMap_;

public:
  AccessIDRemapper(InstantiationType* instantiation, int oldAccessID, int newAccessID,
                   std::unordered_map<std::shared_ptr<Expr>, int>& ExprToAccessIDMap,
                   std::unordered_map<std::shared_ptr<Stmt>, int>& StmtToAccessIDMap)
      : instantiation_(instantiation), oldAccessID_(oldAccessID), newAccessID_(newAccessID),
        ExprToAccessIDMap_(ExprToAccessIDMap), StmtToAccessIDMap_(StmtToAccessIDMap) {}

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    int& varAccessID = StmtToAccessIDMap_[stmt];
    if(varAccessID == oldAccessID_)
      varAccessID = newAccessID_;
    ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    StencilFunctionInstantiation* fun = instantiation_->getStencilFunctionInstantiation(expr);
    fun->renameCallerAccessID(oldAccessID_, newAccessID_);
    ASTVisitorForwarding::visit(expr);
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    int& varAccessID = ExprToAccessIDMap_[expr];
    if(varAccessID == oldAccessID_)
      varAccessID = newAccessID_;
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    int& fieldAccessID = ExprToAccessIDMap_[expr];
    if(fieldAccessID == oldAccessID_)
      fieldAccessID = newAccessID_;
  }
};

/// @brief Remap all accesses from `oldAccessID` to `newAccessID` in the `accessesMap`
static void renameAccessesMaps(std::unordered_map<int, Extents>& accessesMap, int oldAccessID,
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
    StencilInstantiation* instantiation, int oldAccessID, int newAccessID,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs) {
  AccessIDRemapper<StencilInstantiation> remapper(instantiation, oldAccessID, newAccessID,
                                                  instantiation->getExprToAccessIDMap(),
                                                  instantiation->getStmtToAccessIDMap());

  for(auto& statementAccessesPair : statementAccessesPairs)
    statementAccessesPair->getStatement()->ASTStmt->accept(remapper);
}

void renameAccessIDInStmts(
    StencilFunctionInstantiation* instantiation, int oldAccessID, int newAccessID,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs) {
  AccessIDRemapper<StencilFunctionInstantiation> remapper(
      instantiation, oldAccessID, newAccessID, instantiation->getExprToCallerAccessIDMap(),
      instantiation->getStmtToCallerAccessIDMap());

  for(auto& statementAccessesPair : statementAccessesPairs)
    statementAccessesPair->getStatement()->ASTStmt->accept(remapper);
}

void renameAccessIDInExpr(StencilInstantiation* instantiation, int oldAccessID, int newAccessID,
                          std::shared_ptr<Expr>& expr) {
  AccessIDRemapper<StencilInstantiation> remapper(instantiation, oldAccessID, newAccessID,
                                                  instantiation->getExprToAccessIDMap(),
                                                  instantiation->getStmtToAccessIDMap());
  expr->accept(remapper);
}

void renameAccessIDInAccesses(
    StencilInstantiation* instantiation, int oldAccessID, int newAccessID,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs) {
  for(auto& statementAccessesPair : statementAccessesPairs) {
    renameAccessesMaps(statementAccessesPair->getAccesses()->getReadAccesses(), oldAccessID,
                       newAccessID);
    renameAccessesMaps(statementAccessesPair->getAccesses()->getWriteAccesses(), oldAccessID,
                       newAccessID);
  }
}

void renameAccessIDInAccesses(
    StencilFunctionInstantiation* instantiation, int oldAccessID, int newAccessID,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs) {
  for(auto& statementAccessesPair : statementAccessesPairs) {
    renameAccessesMaps(statementAccessesPair->getCallerAccesses()->getReadAccesses(), oldAccessID,
                       newAccessID);
    renameAccessesMaps(statementAccessesPair->getCallerAccesses()->getWriteAccesses(), oldAccessID,
                       newAccessID);
    renameAccessesMaps(statementAccessesPair->getCalleeAccesses()->getReadAccesses(), oldAccessID,
                       newAccessID);
    renameAccessesMaps(statementAccessesPair->getCalleeAccesses()->getWriteAccesses(), oldAccessID,
                       newAccessID);
  }
}

} // namespace dawn
