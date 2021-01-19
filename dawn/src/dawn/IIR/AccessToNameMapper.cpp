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

#include "dawn/IIR/AccessToNameMapper.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/AST/ASTVisitor.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"

namespace dawn {
namespace iir {

void AccessToNameMapper::visit(const std::shared_ptr<ast::VarDeclStmt>& stmt) {
  insertAccessInfo(stmt);
  ast::ASTVisitorForwarding::visit(stmt);
}

void AccessToNameMapper::visit(const std::shared_ptr<ast::StencilFunCallExpr>& expr) {
  if(!curFunctionInstantiation_.empty()) {
    auto* stencilFunctionInstantiation =
        curFunctionInstantiation_.top()->getStencilFunctionInstantiation(expr).get();
    curFunctionInstantiation_.push(stencilFunctionInstantiation);
  } else {
    auto* stencilFunctionInstantiation = metaData_.getStencilFunctionInstantiation(expr).get();
    curFunctionInstantiation_.push(stencilFunctionInstantiation);
  }
  curFunctionInstantiation_.top()->getAST()->accept(*this);

  curFunctionInstantiation_.pop();
  ast::ASTVisitorForwarding::visit(expr);
}

void AccessToNameMapper::insertAccessInfo(const std::shared_ptr<ast::Expr>& expr) {
  int accessID = iir::getAccessID(expr);
  std::string name = (curFunctionInstantiation_.empty())
                         ? metaData_.getNameFromAccessID(accessID)
                         : curFunctionInstantiation_.top()->getNameFromAccessID(accessID);

  accessIDToName_.emplace(accessID, name);
}
void AccessToNameMapper::insertAccessInfo(const std::shared_ptr<ast::VarDeclStmt>& stmt) {
  int accessID = iir::getAccessID(stmt);
  std::string name = (curFunctionInstantiation_.empty())
                         ? metaData_.getNameFromAccessID(accessID)
                         : curFunctionInstantiation_.top()->getNameFromAccessID(accessID);

  accessIDToName_.emplace(accessID, name);
}

void AccessToNameMapper::visit(const std::shared_ptr<ast::VarAccessExpr>& expr) {
  insertAccessInfo(expr);
}

void AccessToNameMapper::visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) {
  insertAccessInfo(expr);
  ASTVisitorForwarding::visit(expr);
}

} // namespace iir
} // namespace dawn
