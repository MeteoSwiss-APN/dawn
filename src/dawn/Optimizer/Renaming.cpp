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
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/Statement.h"
#include <unordered_map>

namespace dawn {

namespace {

/// @brief Remap all accesses from `oldAccessID` to `newAccessID` in all statements
// template <class InstantiationType>
class AccessIDRemapper : public ASTVisitorForwarding {
  //  InstantiationType* instantiation_;

  int oldAccessID_;
  int newAccessID_;

  iir::IIR* iir_;

public:
  AccessIDRemapper(/*InstantiationType* instantiation,*/ int oldAccessID, int newAccessID,
                   iir::IIR* iir)
      : oldAccessID_(oldAccessID), newAccessID_(newAccessID), iir_(iir) {}

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    int varAccessID = iir_->getMetaData()->getAccessIDFromStmt(stmt);
    if(varAccessID == oldAccessID_)
      iir_->getMetaData()->setAccessIDOfStmt(stmt, newAccessID_);
    ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    std::shared_ptr<iir::StencilFunctionInstantiation> fun =
        iir_->getMetaData()->getStencilFunctionInstantiation(expr);
    fun->renameCallerAccessID(oldAccessID_, newAccessID_);
    ASTVisitorForwarding::visit(expr);
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    int varAccessID = iir_->getMetaData()->getAccessIDFromExpr(expr);
    if(varAccessID == oldAccessID_)
      iir_->getMetaData()->setAccessIDOfExpr(expr, newAccessID_);
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    int fieldAccessID = iir_->getMetaData()->getAccessIDFromExpr(expr);
    if(fieldAccessID == oldAccessID_)
      iir_->getMetaData()->setAccessIDOfExpr(expr, newAccessID_);
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
    iir::IIR* iir, int oldAccessID, int newAccessID,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) {
  AccessIDRemapper remapper(/*instantiation,*/ oldAccessID, newAccessID, iir);

  for(auto& statementAccessesPair : statementAccessesPairs)
    statementAccessesPair->getStatement()->ASTStmt->accept(remapper);
}

void renameAccessIDInExpr(iir::IIR* iir, int oldAccessID, int newAccessID,
                          std::shared_ptr<Expr>& expr) {
  AccessIDRemapper remapper(oldAccessID, newAccessID, iir);
  expr->accept(remapper);
}

// void renameAccessIDInAccesses(
//    iir::StencilInstantiation* instantiation, int oldAccessID, int newAccessID,
//    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) {
//  for(auto& statementAccessesPair : statementAccessesPairs) {
//    renameAccessesMaps(statementAccessesPair->getAccesses()->getReadAccesses(), oldAccessID,
//                       newAccessID);
//    renameAccessesMaps(statementAccessesPair->getAccesses()->getWriteAccesses(), oldAccessID,
//                       newAccessID);
//  }
//}

void renameAccessIDInAccesses(
    int oldAccessID, int newAccessID,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) {
  for(auto& statementAccessesPair : statementAccessesPairs) {
    renameAccessesMaps(statementAccessesPair->getCallerAccesses()->getReadAccesses(), oldAccessID,
                       newAccessID);
    renameAccessesMaps(statementAccessesPair->getCallerAccesses()->getWriteAccesses(), oldAccessID,
                       newAccessID);
    if(statementAccessesPair->getCalleeAccesses()) {
      renameAccessesMaps(statementAccessesPair->getCalleeAccesses()->getReadAccesses(), oldAccessID,
                         newAccessID);
      renameAccessesMaps(statementAccessesPair->getCalleeAccesses()->getWriteAccesses(),
                         oldAccessID, newAccessID);
    }
  }
}

} // namespace dawn
