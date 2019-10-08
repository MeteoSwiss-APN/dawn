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
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"

namespace dawn {
namespace iir {

void AccessToNameMapper::visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) {
  insertAccessInfo(stmt);
  iir::ASTVisitorForwarding::visit(stmt);
}

void AccessToNameMapper::visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
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
  iir::ASTVisitorForwarding::visit(expr);
}

void AccessToNameMapper::insertAccessInfo(const std::shared_ptr<iir::Expr>& expr) {
  int accessID = iir::getAccessIDFromExpr(expr);
  std::string name = (curFunctionInstantiation_.empty())
                         ? metaData_.getNameFromAccessID(accessID)
                         : curFunctionInstantiation_.top()->getNameFromAccessID(accessID);

  accessIDToName_.emplace(accessID, name);
}
void AccessToNameMapper::insertAccessInfo(const std::shared_ptr<iir::VarDeclStmt>& stmt) {
  int accessID = iir::getAccessIDFromStmt(stmt);
  std::string name = (curFunctionInstantiation_.empty())
                         ? metaData_.getNameFromAccessID(accessID)
                         : curFunctionInstantiation_.top()->getNameFromAccessID(accessID);

  accessIDToName_.emplace(accessID, name);
}

void AccessToNameMapper::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {
  insertAccessInfo(expr);
}

void AccessToNameMapper::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  insertAccessInfo(expr);
}

} // namespace iir
} // namespace dawn
