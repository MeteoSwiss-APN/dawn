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

#include "PassSimplifyStatements.h"
#include "dawn/AST/ASTExpr.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Type.h"
#include <memory>

namespace dawn {

namespace {
class IncrementDecrementReplacer : public ast::ASTVisitorPostOrder {
  std::vector<std::shared_ptr<ast::Stmt>> statements_;

public:
  std::shared_ptr<iir::Expr>
  postVisitNode(std::shared_ptr<iir::UnaryOperator> const& unaryOp) override {
    auto sourceLoc = unaryOp->getSourceLocation();
    std::shared_ptr<ast::BinaryOperator> binOp;
    if(unaryOp->getOp() == "++") {
      binOp = std::make_shared<ast::BinaryOperator>(
          unaryOp->getOperand()->clone(), "+",
          std::make_shared<ast::LiteralAccessExpr>("1", BuiltinTypeID::Integer, sourceLoc),
          sourceLoc);
    } else if(unaryOp->getOp() == "--") {
      binOp = std::make_shared<ast::BinaryOperator>(
          unaryOp->getOperand()->clone(), "-",
          std::make_shared<ast::LiteralAccessExpr>("1", BuiltinTypeID::Integer, sourceLoc),
          sourceLoc);
    } else {
      return unaryOp;
    }
    DAWN_ASSERT(unaryOp->getOperand()->getKind() == ast::Expr::Kind::FieldAccessExpr ||
                unaryOp->getOperand()->getKind() == ast::Expr::Kind::VarAccessExpr);
    auto newAssignmentExpr = std::make_shared<ast::AssignmentExpr>(unaryOp->getOperand()->clone(),
                                                                   binOp, "=", sourceLoc);

    statements_.push_back(iir::makeExprStmt(newAssignmentExpr, sourceLoc));

    return unaryOp->getOperand();
  }
  const std::vector<std::shared_ptr<ast::Stmt>>& getReplacements() { return statements_; }
};
} // namespace

bool PassSimplifyStatements::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilInstantiation->getIIR())) {
    for(auto stmtIt = doMethod->getAST().getStatements().begin();
        stmtIt != doMethod->getAST().getStatements().end();) {
      // Compound assignment
      if(const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(*stmtIt)) {
        auto sourceLoc = exprStmt->getSourceLocation();
        if(const auto& assignmentExpr =
               std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())) {
          if(assignmentExpr->getOp() != "=") {
            auto binOp = std::make_shared<ast::BinaryOperator>(
                assignmentExpr->getLeft()->clone(), assignmentExpr->getOp().substr(0, 1),
                assignmentExpr->getRight(), sourceLoc);
            auto newAssignmentExpr = std::make_shared<ast::AssignmentExpr>(
                assignmentExpr->getLeft(), binOp, "=", sourceLoc);
            exprStmt->getExpr() = newAssignmentExpr;
          }
        }
      }

      // Increment/decrement ops (can be nested inside expression tree)
      IncrementDecrementReplacer replacer;
      doMethod->getAST().substitute(stmtIt, (*stmtIt)->acceptAndReplace(replacer));
      stmtIt = doMethod->getAST().insert(stmtIt, replacer.getReplacements().begin(),
                                         replacer.getReplacements().end());
      std::advance(stmtIt, replacer.getReplacements().size());
      // Substitution might have left an useless statement accessing a field/variable.
      if(const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(*stmtIt)) {
        if(exprStmt->getExpr()->getKind() == ast::Expr::Kind::FieldAccessExpr ||
           exprStmt->getExpr()->getKind() == ast::Expr::Kind::VarAccessExpr) {
          stmtIt = doMethod->getAST().erase(stmtIt);
          continue;
        }
      }

      ++stmtIt;
    }
  }
  return true;
}

} // namespace dawn
