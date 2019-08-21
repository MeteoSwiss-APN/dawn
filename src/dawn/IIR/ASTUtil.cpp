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
#include "dawn/SIR/Statement.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/Unreachable.h"
#include <functional>

namespace dawn {
namespace iir {

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

/// @brief helper to find all the fields in a statement
class FieldFinder : public iir::ASTVisitorForwarding {
public:
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) {
    auto fieldFromExpression = sir::Field(expr->getName());
    auto iter = std::find(allFields_.begin(), allFields_.end(), fieldFromExpression);
    if(iter == allFields_.end())
      allFields_.push_back(fieldFromExpression);
    iir::ASTVisitorForwarding::visit(expr);
  }

  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
    stmt->getVerticalRegion()->Ast->accept(*this);
  }

  const std::vector<sir::Field>& getFields() const { return allFields_; }

private:
  std::vector<sir::Field> allFields_;
};

extern std::vector<sir::Field> getFieldFromStencilAST(const std::shared_ptr<AST>& ast) {
  FieldFinder finder;
  ast->accept(finder);
  return finder.getFields();
}

} // namespace iir
} // namespace dawn
