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
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {
namespace iir {

void AccessToNameMapper::visit(const std::shared_ptr<VarDeclStmt>& stmt) {
  insertAccessInfo(stmt);
  ASTVisitorForwarding::visit(stmt);
}

void AccessToNameMapper::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  if(!curFunctionInstantiation_.empty()) {
    auto* stencilFunctionInstantiation =
        curFunctionInstantiation_.top()->getStencilFunctionInstantiation(expr).get();
    curFunctionInstantiation_.push(stencilFunctionInstantiation);
  } else {
    auto* stencilFunctionInstantiation =
        stencilInstantiation_->getStencilFunctionInstantiation(expr).get();
    curFunctionInstantiation_.push(stencilFunctionInstantiation);
  }
  curFunctionInstantiation_.top()->getAST()->accept(*this);

  curFunctionInstantiation_.pop();
  ASTVisitorForwarding::visit(expr);
}

void AccessToNameMapper::insertAccessInfo(const std::shared_ptr<Expr>& expr) {
  int accessID = (curFunctionInstantiation_.empty())
                     ? stencilInstantiation_->getAccessIDFromExpr(expr)
                     : curFunctionInstantiation_.top()->getAccessIDFromExpr(expr);
  std::string name = (curFunctionInstantiation_.empty())
                         ? stencilInstantiation_->getNameFromAccessID(accessID)
                         : curFunctionInstantiation_.top()->getNameFromAccessID(accessID);

  accessIDToName_.emplace(accessID, name);
}
void AccessToNameMapper::insertAccessInfo(const std::shared_ptr<Stmt>& stmt) {
  int accessID = (curFunctionInstantiation_.empty())
                     ? stencilInstantiation_->getAccessIDFromStmt(stmt)
                     : curFunctionInstantiation_.top()->getAccessIDFromStmt(stmt);
  std::string name = (curFunctionInstantiation_.empty())
                         ? stencilInstantiation_->getNameFromAccessID(accessID)
                         : curFunctionInstantiation_.top()->getNameFromAccessID(accessID);

  accessIDToName_.emplace(accessID, name);
}

void AccessToNameMapper::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  insertAccessInfo(expr);
}

void AccessToNameMapper::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  insertAccessInfo(expr);
}

} // namespace iir
} // namespace dawn
