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
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/Statement.h"
#include <unordered_map>

namespace dawn {

namespace {

/// @brief Get all field and variable accesses identifier by `AccessID`
class GetFieldAndVarAccesses : public ASTVisitorForwarding {
  const iir::IIR* iir_;
  int AccessID_;

  std::vector<std::shared_ptr<FieldAccessExpr>> fieldAccessExprToReplace_;
  std::vector<std::shared_ptr<VarAccessExpr>> varAccessesToReplace_;

public:
  GetFieldAndVarAccesses(const iir::IIR* iir, int AccessID) : iir_(iir), AccessID_(AccessID) {
    //    std::cout << "map after ctor" << metaInfo_->getExprToAccessIDMap().size() << std::endl;
    //    std::cout << "and og map size " << metaInfo->getExprToAccessIDMap().size() << std::endl;
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    //    std::cout << "visit varaccess, mapsize outside:" <<
    //    metaInfo_->getExprToAccessIDMap().size()
    //              << std::endl;
    //    if(metaInfo_->getAccessIDFromExpr(expr) == AccessID_)
    //      varAccessesToReplace_.emplace_back(expr);
    if(iir_->getMetaData()->getAccessIDFromExpr(expr) == AccessID_)
      varAccessesToReplace_.emplace_back(expr);
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    //    std::cout << "visit FieldAccessExpr, mapsize outside:"
    //              << metaInfo_->getExprToAccessIDMap().size() << std::endl;
    //    if(metaInfo_->getAccessIDFromExpr(expr) == AccessID_)
    //      fieldAccessExprToReplace_.emplace_back(expr);
    if(iir_->getMetaData()->getAccessIDFromExpr(expr) == AccessID_)
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
    iir::Stencil* stencil, int AccessID, const std::string& varname,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) {
  //  iir::StencilInstantiation& instantiation = stencil->getStencilInstantiation();
  iir::IIR* internalIR = stencil->getParent().get();

  GetFieldAndVarAccesses visitor(internalIR, AccessID);
  for(const auto& statementAccessesPair : statementAccessesPairs) {
    visitor.reset();

    const auto& stmt = statementAccessesPair->getStatement()->ASTStmt;
    stmt->accept(visitor);

    for(auto& oldExpr : visitor.getFieldAccessExprToReplace()) {
      auto newExpr = std::make_shared<VarAccessExpr>(varname);

      replaceOldExprWithNewExprInStmt(stmt, oldExpr, newExpr);

      internalIR->getMetaData()->mapExprToAccessID(newExpr, AccessID);
      internalIR->getMetaData()->eraseExprToAccessID(oldExpr);
    }
  }
}

void replaceVarWithFieldAccessInStmts(
    iir::Stencil* stencil, int AccessID, const std::string& fieldname,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs) {
  //  const auto& iir = stencil->getParent(); // ->getStencilInstantiation();
  iir::IIR* internalIR = stencil->getParent().get();

//  std::cout << "size here: " << internalIR->getMetaData()->getExprToAccessIDMap().size()
//            << std::endl;

  GetFieldAndVarAccesses visitor(internalIR, AccessID);
  for(const auto& statementAccessesPair : statementAccessesPairs) {
    visitor.reset();

    const auto& stmt = statementAccessesPair->getStatement()->ASTStmt;
    stmt->accept(visitor);

    for(auto& oldExpr : visitor.getVarAccessesToReplace()) {
      auto newExpr = std::make_shared<FieldAccessExpr>(fieldname);

      replaceOldExprWithNewExprInStmt(stmt, oldExpr, newExpr);

      internalIR->getMetaData()->mapExprToAccessID(newExpr, AccessID);
      internalIR->getMetaData()->eraseExprToAccessID(oldExpr);
    }
  }
}

namespace {

/// @brief Get all field and variable accesses identifier by `AccessID`
class GetStencilCalls : public ASTVisitorForwarding {
  const std::shared_ptr<iir::StencilMetaInformation>& metaInfo_;
  int StencilID_;

  std::vector<std::shared_ptr<StencilCallDeclStmt>> stencilCallsToReplace_;

public:
  GetStencilCalls(const std::shared_ptr<iir::StencilMetaInformation>& metaInfo, int StencilID)
      : metaInfo_(metaInfo), StencilID_(StencilID) {}

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    if(metaInfo_->getStencilIDFromStmt(stmt) == StencilID_)
      stencilCallsToReplace_.emplace_back(stmt);
  }

  std::vector<std::shared_ptr<StencilCallDeclStmt>>& getStencilCallsToReplace() {
    return stencilCallsToReplace_;
  }

  void reset() { stencilCallsToReplace_.clear(); }
};

} // anonymous namespace

void replaceStencilCalls(iir::IIR* iir, int oldStencilID, const std::vector<int>& newStencilIDs) {
  GetStencilCalls visitor(iir->getMetaData(), oldStencilID);

  for(auto& statement : iir->getMetaData()->getStencilDescStatements()) {
    visitor.reset();

    std::shared_ptr<Stmt>& stmt = statement->ASTStmt;

    stmt->accept(visitor);
    for(auto& oldStencilCall : visitor.getStencilCallsToReplace()) {

      // Create the new stencils
      std::vector<std::shared_ptr<StencilCallDeclStmt>> newStencilCalls;
      for(int StencilID : newStencilIDs) {
        auto placeholderStencil = std::make_shared<sir::StencilCall>(
            iir::StencilMetaInformation::makeStencilCallCodeGenName(StencilID));
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

      iir->getMetaData()->getStencilCallToStencilIDMap().erase(oldStencilCall);
      for(std::size_t i = 0; i < newStencilIDs.size(); ++i) {
        iir->getMetaData()->getStencilCallToStencilIDMap().emplace(newStencilCalls[i],
                                                                   newStencilIDs[i]);
        iir->getMetaData()->getIDToStencilCallMap().emplace(newStencilIDs[i], newStencilCalls[i]);
      }
    }
  }
}

} // namespace dawn
