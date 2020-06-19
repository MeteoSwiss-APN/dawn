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

#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/Unreachable.h"
#include <functional>

namespace dawn {
namespace iir {
//
// TODO refactor_AST: this is TEMPORARY, will be removed in the future
//
void replaceOldExprWithNewExprInStmt(const std::shared_ptr<Stmt>& stmt,
                                     const std::shared_ptr<Expr>& oldExpr,
                                     const std::shared_ptr<Expr>& newExpr) {
  ast::replaceOldExprWithNewExprInStmt(stmt, oldExpr, newExpr);
}

void replaceOldStmtWithNewStmtInStmt(const std::shared_ptr<Stmt>& stmt,
                                     const std::shared_ptr<Stmt>& oldStmt,
                                     const std::shared_ptr<Stmt>& newStmt) {
  ast::replaceOldStmtWithNewStmtInStmt(stmt, oldStmt, newStmt);
}

bool evalExprAsDouble(const std::shared_ptr<Expr>& expr, double& result,
                      const std::unordered_map<std::string, double>& variableMap) {
  return ast::evalExprAsDouble(expr, result, variableMap);
}
bool evalExprAsInteger(const std::shared_ptr<Expr>& expr, int& result,
                       const std::unordered_map<std::string, double>& variableMap) {
  return ast::evalExprAsInteger(expr, result, variableMap);
}
bool evalExprAsBoolean(const std::shared_ptr<Expr>& expr, bool& result,
                       const std::unordered_map<std::string, double>& variableMap) {
  return ast::evalExprAsBoolean(expr, result, variableMap);
}

} // namespace iir
} // namespace dawn
