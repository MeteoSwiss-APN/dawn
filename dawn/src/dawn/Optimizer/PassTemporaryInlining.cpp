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
#include <memory>
#include <unordered_map>

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

class FieldAccessesReplacer : public ast::ASTVisitorPostOrder {
  const int fieldAccessToSubstitute_;
  const std::shared_ptr<const ast::Expr> substituteExpr_;

public:
  std::shared_ptr<ast::Expr>
  postVisitNode(std::shared_ptr<ast::FieldAccessExpr> const& expr) override {
    if(iir::getAccessID(expr) == fieldAccessToSubstitute_) {
      return substituteExpr_->clone();
    }
    return expr;
  }

  FieldAccessesReplacer(int fieldAccessToSubstitute,
                        const std::shared_ptr<const ast::Expr> substituteExpr)
      : fieldAccessToSubstitute_(fieldAccessToSubstitute), substituteExpr_(substituteExpr) {}
};

bool PassTemporaryInlining::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {

  dawn::iir::ASTMatcher asgnMatcher(stencilInstantiation.get());
  std::vector<std::shared_ptr<ast::Expr>>& assignmentsExprs =
      asgnMatcher.match(ast::Expr::Kind::AssignmentExpr);

  std::unordered_map<int, int> occCount;
  for(const auto& assignmentsExpr : assignmentsExprs) {
    const auto& cand = std::static_pointer_cast<ast::BinaryOperator>(assignmentsExpr);
    if(auto lhs = std::dynamic_pointer_cast<ast::FieldAccessExpr>(cand->getLeft())) {
      occCount[iir::getAccessID(lhs)]++;
    }
  }

  std::vector<std::shared_ptr<ast::AssignmentExpr>> candidates;
  for(const auto& it : assignmentsExprs) {
    const auto& cand = std::static_pointer_cast<ast::BinaryOperator>(it);
    if(cand->getLeft()->getKind() == ast::Expr::Kind::FieldAccessExpr &&
       cand->getRight()->getKind() == ast::Expr::Kind::ReductionOverNeighborExpr) {
      if(auto lhs = std::dynamic_pointer_cast<ast::FieldAccessExpr>(cand->getLeft())) {
        if(stencilInstantiation->getMetaData().isAccessType(iir::FieldAccessType::StencilTemporary,
                                                            iir::getAccessID(lhs)) &&
           occCount.at(iir::getAccessID(lhs)) == 1) {
          candidates.push_back(std::static_pointer_cast<ast::AssignmentExpr>(cand));

          DAWN_LOG(INFO) << stencilInstantiation->getName() << " inlined computation of "
                         << stencilInstantiation->getMetaData().getNameFromAccessID(
                                iir::getAccessID(lhs));
        }
      }
    }
  }

  for(auto cand : candidates) {
    int fieldAccess =
        iir::getAccessID(std::static_pointer_cast<ast::FieldAccessExpr>(cand->getLeft()));
    FieldAccessesReplacer replacer(fieldAccess, cand->getRight());
    for(auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilInstantiation->getIIR())) {
      for(auto stmtIt = doMethod->getAST().getStatements().begin();
          stmtIt != doMethod->getAST().getStatements().end();) {

        if((*stmtIt)->getKind() == ast::Stmt::Kind::ExprStmt) {
          const auto& exprStmt = std::static_pointer_cast<ast::ExprStmt>(*stmtIt);
          if(const auto& binaryOp =
                 std::dynamic_pointer_cast<ast::BinaryOperator>(exprStmt->getExpr())) {
            if(iir::getAccessID(binaryOp->getLeft()) == fieldAccess) {
              doMethod->getAST().erase(stmtIt);
              continue;
            }
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
    int fieldAccess =
        iir::getAccessID(std::static_pointer_cast<ast::FieldAccessExpr>(cand->getLeft()));
    stencilInstantiation->getMetaData().removeAccessID(fieldAccess);
  }

  cleanUp(stencilInstantiation);

  return true;
}

} // namespace dawn