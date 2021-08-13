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

#include "dawn/Optimizer/PassTemporaryInlining.h"
#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTMatcher.h"
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/Unreachable.h"
#include <memory>
#include <unordered_map>
#include <variant>

namespace {
void cleanUp(const std::shared_ptr<dawn::iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& multiStage :
      dawn::iterateIIROver<dawn::iir::MultiStage>(*stencilInstantiation->getIIR())) {
    for(auto curStageIt = multiStage->childrenBegin(); curStageIt != multiStage->childrenEnd();
        curStageIt++) {
      dawn::iir::Stage& curStage = **curStageIt;
      for(auto curDoMethodIt = curStage.childrenBegin(); curDoMethodIt != curStage.childrenEnd();) {
        dawn::iir::DoMethod& curDoMethod = **curDoMethodIt;

        if(curDoMethod.isEmptyOrNullStmt()) {
          DAWN_LOG(INFO) << stencilInstantiation->getName() << ": DoMethod: " << curDoMethod.getID()
                         << " has empty body after inlining a temporary, removing";

          curDoMethodIt = curStage.childrenErase(curDoMethodIt);
        } else {
          curDoMethodIt++;
        }
      }

      for(auto& doMethod : curStage.getChildren()) {
        doMethod->update(dawn::iir::NodeUpdateType::level);
      }
      curStage.update(dawn::iir::NodeUpdateType::levelAndTreeAbove);
    }

    for(auto curStageIt = multiStage->childrenBegin(); curStageIt != multiStage->childrenEnd();) {
      dawn::iir::Stage& curStage = **curStageIt;
      if(curStage.childrenEmpty()) {
        curStageIt = multiStage->childrenErase(curStageIt);
      } else {
        curStageIt++;
      }
    }
  }
}
} // namespace

namespace dawn {

class AccessesReplacer : public ast::ASTVisitorPostOrder {
  const int accessToSubstitute_;
  const std::shared_ptr<const ast::Expr> substituteExpr_;

public:
  std::shared_ptr<ast::Expr>
  postVisitNode(std::shared_ptr<ast::FieldAccessExpr> const& expr) override {
    if(iir::getAccessID(expr) == accessToSubstitute_) {
      return substituteExpr_->clone();
    }
    return expr;
  }

  std::shared_ptr<ast::Expr>
  postVisitNode(std::shared_ptr<ast::VarAccessExpr> const& expr) override {
    if(iir::getAccessID(expr) == accessToSubstitute_) {
      return substituteExpr_->clone();
    }
    return expr;
  }

  AccessesReplacer(int accessToSubstitute, const std::shared_ptr<const ast::Expr> substituteExpr)
      : accessToSubstitute_(accessToSubstitute), substituteExpr_(substituteExpr) {}
};

bool PassTemporaryInlining::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {

  using candidate_t =
      std::variant<std::shared_ptr<ast::AssignmentExpr>, std::shared_ptr<ast::VarDeclStmt>>;

  dawn::iir::ASTMatcher assignMatcher(stencilInstantiation.get());
  dawn::iir::ASTMatcher varDeclMatcher(stencilInstantiation.get());
  std::vector<std::shared_ptr<ast::Expr>>& assignmentsExprs =
      assignMatcher.match(ast::Expr::Kind::AssignmentExpr);
  std::vector<std::shared_ptr<ast::Stmt>>& varDeclStmts =
      varDeclMatcher.match(ast::Stmt::Kind::VarDeclStmt);

  std::unordered_map<int, int> occCount;
  for(const auto& varDeclStmtIt : varDeclStmts) {
    const auto& cand = std::static_pointer_cast<ast::VarDeclStmt>(varDeclStmtIt);
    occCount[iir::getAccessID(cand)]++;
  }
  for(const auto& assignmentsExpr : assignmentsExprs) {
    const auto& cand = std::static_pointer_cast<ast::BinaryOperator>(assignmentsExpr);
    if(auto lhs = std::dynamic_pointer_cast<ast::FieldAccessExpr>(cand->getLeft())) {
      occCount[iir::getAccessID(lhs)]++;
    }
    if(auto lhs = std::dynamic_pointer_cast<ast::VarAccessExpr>(cand->getLeft())) {
      occCount[iir::getAccessID(lhs)]++;
    }
  }

  std::vector<candidate_t> candidates;
  for(const auto& it : assignmentsExprs) {
    const auto& cand = std::static_pointer_cast<ast::BinaryOperator>(it);
    if(auto lhs = std::dynamic_pointer_cast<ast::FieldAccessExpr>(cand->getLeft())) {
      if(stencilInstantiation->getMetaData().isAccessType(iir::FieldAccessType::StencilTemporary,
                                                          iir::getAccessID(lhs)) &&
         ast::dimension_cast<const ast::UnstructuredFieldDimension&>(
             stencilInstantiation->getMetaData()
                 .getFieldDimensions(iir::getAccessID(lhs))
                 .getHorizontalFieldDimension())
             .isDense() &&
         occCount.at(iir::getAccessID(lhs)) == 1) {
        candidates.push_back(std::static_pointer_cast<ast::AssignmentExpr>(cand));
        DAWN_LOG(INFO) << stencilInstantiation->getName()
                       << " inlined computation of temporary Field"
                       << stencilInstantiation->getMetaData().getNameFromAccessID(
                              iir::getAccessID(lhs));
      }
    }
  }
  for(const auto& varDeclStmtIt : varDeclStmts) {
    const auto& cand = std::static_pointer_cast<ast::VarDeclStmt>(varDeclStmtIt);
    if(occCount.at(iir::getAccessID(cand)) == 1 && cand->getInitList().size() == 1) {
      candidates.push_back(cand);
      DAWN_LOG(INFO) << stencilInstantiation->getName() << " inlined computation of var "
                     << stencilInstantiation->getMetaData().getNameFromAccessID(
                            iir::getAccessID(cand));
    }
  }

  auto getFieldOrVarAccessID = [](const candidate_t cand) -> int {
    if(auto assignment = std::get_if<std::shared_ptr<ast::AssignmentExpr>>(&cand)) {
      auto lhs = std::static_pointer_cast<ast::AssignmentExpr>(assignment->get()->getLeft());
      return iir::getAccessID(lhs);
    }
    if(auto varDecl = std::get_if<std::shared_ptr<ast::VarDeclStmt>>(&cand)) {
      return iir::getAccessID(*varDecl);
    }
    dawn_unreachable("invalid candidate");
  };

  auto getRhs = [](const candidate_t cand) -> std::shared_ptr<ast::Expr> {
    if(auto assignment = std::get_if<std::shared_ptr<ast::AssignmentExpr>>(&cand)) {
      return assignment->get()->getRight();
    }
    if(const auto& varDecl = std::get_if<std::shared_ptr<ast::VarDeclStmt>>(&cand)) {
      DAWN_ASSERT(varDecl->get()->getInitList().size() == 1);
      return varDecl->get()->getInitList()[0];
    }
    dawn_unreachable("invalid candidate");
  };

  for(auto cand : candidates) {
    int accessID = getFieldOrVarAccessID(cand);
    AccessesReplacer replacer(accessID, getRhs(cand));

    for(auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilInstantiation->getIIR())) {
      for(auto stmtIt = doMethod->getAST().getStatements().begin();
          stmtIt != doMethod->getAST().getStatements().end();) {

        if((*stmtIt)->getKind() == ast::Stmt::Kind::ExprStmt) {
          const auto& exprStmt = std::static_pointer_cast<ast::ExprStmt>(*stmtIt);
          if(const auto& binaryOp =
                 std::dynamic_pointer_cast<ast::BinaryOperator>(exprStmt->getExpr())) {
            if(iir::getAccessID(binaryOp->getLeft()) == accessID) {
              doMethod->getAST().erase(stmtIt);
              continue;
            }
          }
        }

        if((*stmtIt)->getKind() == ast::Stmt::Kind::VarDeclStmt) {
          const auto& varDeclStmt = std::static_pointer_cast<ast::VarDeclStmt>(*stmtIt);
          if(iir::getAccessID(varDeclStmt) == accessID) {
            doMethod->getAST().erase(stmtIt);
            continue;
          }
        }

        (*stmtIt)->acceptAndReplace(replacer);
        stmtIt++;
      }

      computeAccesses(stencilInstantiation->getMetaData(), doMethod->getAST().getStatements());
      doMethod->update(iir::NodeUpdateType::levelAndTreeAbove);
    }
  }

  for(auto cand : candidates) {
    int accessID = getFieldOrVarAccessID(cand);
    stencilInstantiation->getMetaData().removeAccessID(accessID);
  }

  cleanUp(stencilInstantiation);

  return true;
}

} // namespace dawn