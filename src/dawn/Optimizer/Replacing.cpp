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

#include "dawn/Optimizer/Replacing.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/Statement.h"
#include <unordered_map>

namespace dawn {

namespace {

/// @brief Get all field and variable accesses identifier by `AccessID`
class GetFieldAndVarAccesses : public ASTVisitorForwarding {
  StencilInstantiation& instantiation_;
  int AccessID_;

  std::vector<std::shared_ptr<FieldAccessExpr>> fieldAccessExprToReplace_;
  std::vector<std::shared_ptr<VarAccessExpr>> varAccessesToReplace_;

public:
  GetFieldAndVarAccesses(StencilInstantiation& instantiation, int AccessID)
      : instantiation_(instantiation), AccessID_(AccessID) {}

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    if(instantiation_.getAccessIDFromExpr(expr) == AccessID_)
      varAccessesToReplace_.emplace_back(expr);
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    if(instantiation_.getAccessIDFromExpr(expr) == AccessID_)
      fieldAccessExprToReplace_.emplace_back(expr);
  }

  std::vector<std::shared_ptr<VarAccessExpr>>& getVarAccessesToReplace() {
    return varAccessesToReplace_;
  }

  std::vector<std::shared_ptr<FieldAccessExpr>>& getFieldAccessExprToReplace() {
    return fieldAccessExprToReplace_;
  }

  void reset() {
    fieldAccessExprToReplace_.clear();
    varAccessesToReplace_.clear();
  }
};

} // anonymous namespace

void replaceFieldWithVarAccessInStmts(
    Stencil* stencil, int AccessID, const std::string& varname,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs) {
  StencilInstantiation& instantiation = stencil->getStencilInstantiation();

  GetFieldAndVarAccesses visitor(instantiation, AccessID);
  for(const auto& statementAccessesPair : statementAccessesPairs) {
    visitor.reset();

    const auto& stmt = statementAccessesPair->getStatement()->ASTStmt;
    stmt->accept(visitor);

    for(auto& oldExpr : visitor.getFieldAccessExprToReplace()) {
      auto newExpr = std::make_shared<VarAccessExpr>(varname);

      replaceOldExprWithNewExprInStmt(stmt, oldExpr, newExpr);

      instantiation.mapExprToAccessID(newExpr, AccessID);
      instantiation.eraseExprToAccessID(oldExpr);
    }
  }
}

void replaceVarWithFieldAccessInStmts(
    Stencil* stencil, int AccessID, const std::string& fieldname,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs) {
  StencilInstantiation& instantiation = stencil->getStencilInstantiation();

  GetFieldAndVarAccesses visitor(instantiation, AccessID);
  for(const auto& statementAccessesPair : statementAccessesPairs) {
    visitor.reset();

    const auto& stmt = statementAccessesPair->getStatement()->ASTStmt;
    stmt->accept(visitor);

    for(auto& oldExpr : visitor.getVarAccessesToReplace()) {
      auto newExpr = std::make_shared<FieldAccessExpr>(fieldname);

      replaceOldExprWithNewExprInStmt(stmt, oldExpr, newExpr);

      instantiation.mapExprToAccessID(newExpr, AccessID);
      instantiation.eraseExprToAccessID(oldExpr);
    }
  }
}

namespace {

/// @brief Get all field and variable accesses identifier by `AccessID`
class GetStencilCalls : public ASTVisitorForwarding {
  const std::shared_ptr<StencilInstantiation>& instantiation_;
  int StencilID_;

  std::vector<std::shared_ptr<StencilCallDeclStmt>> stencilCallsToReplace_;

public:
  GetStencilCalls(const std::shared_ptr<StencilInstantiation>& instantiation, int StencilID)
      : instantiation_(instantiation), StencilID_(StencilID) {}

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    if(instantiation_->getStencilIDFromStmt(stmt) == StencilID_)
      stencilCallsToReplace_.emplace_back(stmt);
  }

  std::vector<std::shared_ptr<StencilCallDeclStmt>>& getStencilCallsToReplace() {
    return stencilCallsToReplace_;
  }

  void reset() { stencilCallsToReplace_.clear(); }
};

} // anonymous namespace

void replaceStencilCalls(const std::shared_ptr<StencilInstantiation>& instantiation,
                         int oldStencilID, const std::vector<int>& newStencilIDs) {
  GetStencilCalls visitor(instantiation, oldStencilID);

  for(auto& statement : instantiation->getStencilDescStatements()) {
    visitor.reset();

    std::shared_ptr<Stmt>& stmt = statement->ASTStmt;

    stmt->accept(visitor);
    for(auto& oldStencilCall : visitor.getStencilCallsToReplace()) {

      // Create the new stencils
      std::vector<std::shared_ptr<StencilCallDeclStmt>> newStencilCalls;
      for(int StencilID : newStencilIDs) {
        auto placeholderStencil = std::make_shared<sir::StencilCall>(
            StencilInstantiation::makeStencilCallCodeGenName(StencilID));
        newStencilCalls.push_back(std::make_shared<StencilCallDeclStmt>(placeholderStencil));
      }

      // Bundle all the statements in a block statements
      auto newBlockStmt = std::make_shared<BlockStmt>();
      std::copy(newStencilCalls.begin(), newStencilCalls.end(),
                std::back_inserter(newBlockStmt->getStatements()));

      if(oldStencilCall == stmt) {
        // Replace the the statement directly
        DAWN_ASSERT(visitor.getStencilCallsToReplace().size() == 1);
        stmt = newBlockStmt;
      } else {
        // Recursively replace the statement
        replaceOldStmtWithNewStmtInStmt(stmt, oldStencilCall, newBlockStmt);
      }

      instantiation->getStencilCallToStencilIDMap().erase(oldStencilCall);
      for(std::size_t i = 0; i < newStencilIDs.size(); ++i) {
        instantiation->getStencilCallToStencilIDMap().emplace(newStencilCalls[i], newStencilIDs[i]);
        instantiation->getIDToStencilCallMap().emplace(newStencilIDs[i], newStencilCalls[i]);
      }
    }
  }
}

} // namespace dawn
