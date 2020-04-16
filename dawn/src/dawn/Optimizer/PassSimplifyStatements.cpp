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

bool PassSimplifyStatements::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilInstantiation->getIIR())) {
    for(auto stmtIt = doMethod->getAST().getStatements().begin();
        stmtIt != doMethod->getAST().getStatements().end(); ++stmtIt) {
      if(const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(*stmtIt)) {
        auto sourceLoc = exprStmt->getSourceLocation();
        if(const auto& assignmentExpr =
               std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())) {
          if(assignmentExpr->getOp() != "=") { // Compound assignment
            auto binOp = std::make_shared<ast::BinaryOperator>(
                assignmentExpr->getLeft()->clone(), assignmentExpr->getOp().substr(0, 1),
                assignmentExpr->getRight(), sourceLoc);
            auto newAssignmentExpr = std::make_shared<ast::AssignmentExpr>(
                assignmentExpr->getLeft(), binOp, "=", sourceLoc);
            exprStmt->getExpr() = newAssignmentExpr;
          }
          // TODO: rewrite: unary op can be nested inside an expression
        } else if(const auto& unaryOp = std::dynamic_pointer_cast<iir::UnaryOperator>(
                      exprStmt->getExpr())) { // Increment / decrement ops
          if(unaryOp->getOp() == "++") {
            auto binOp = std::make_shared<ast::BinaryOperator>(
                unaryOp->getOperand()->clone(), "+",
                std::make_shared<ast::LiteralAccessExpr>("1", BuiltinTypeID::Integer, sourceLoc),
                sourceLoc);
            auto newAssignmentExpr =
                std::make_shared<ast::AssignmentExpr>(unaryOp->getOperand(), binOp, "=", sourceLoc);
            exprStmt->getExpr() = newAssignmentExpr;
          } else if(unaryOp->getOp() == "--") {
            auto binOp = std::make_shared<ast::BinaryOperator>(
                unaryOp->getOperand()->clone(), "-",
                std::make_shared<ast::LiteralAccessExpr>("1", BuiltinTypeID::Integer, sourceLoc),
                sourceLoc);
            auto newAssignmentExpr =
                std::make_shared<ast::AssignmentExpr>(unaryOp->getOperand(), binOp, "=", sourceLoc);
            exprStmt->getExpr() = newAssignmentExpr;
          }
        }
      }
    }
  }
  return true;
}

} // namespace dawn
