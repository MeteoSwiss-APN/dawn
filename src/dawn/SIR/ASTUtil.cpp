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

#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"

namespace dawn {
namespace sir {

class ExprReplacer : public ast::ExprReplacer<SIRASTData>, public ASTVisitor {
public:
  ExprReplacer(const std::shared_ptr<Expr>& oldExpr, const std::shared_ptr<Expr>& newExpr)
      : ast::ExprReplacer<SIRASTData>(oldExpr, newExpr) {}
  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {}
};

void replaceOldExprWithNewExprInStmt(const std::shared_ptr<Stmt>& stmt,
                                     const std::shared_ptr<Expr>& oldExpr,
                                     const std::shared_ptr<Expr>& newExpr) {
  ExprReplacer replacer(oldExpr, newExpr);
  stmt->accept(replacer);
}

class StmtReplacer : public ast::StmtReplacer<SIRASTData>, public ASTVisitorForwarding {
public:
  StmtReplacer(const std::shared_ptr<Stmt>& oldStmt, const std::shared_ptr<Stmt>& newStmt)
      : ast::StmtReplacer<SIRASTData>(oldStmt, newStmt) {}
};

void replaceOldStmtWithNewStmtInStmt(const std::shared_ptr<Stmt>& stmt,
                                     const std::shared_ptr<Stmt>& oldStmt,
                                     const std::shared_ptr<Stmt>& newStmt) {
  StmtReplacer replacer(oldStmt, newStmt);
  stmt->accept(replacer);
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
class FieldFinder : public ASTVisitorForwarding {
public:
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) {
    auto fieldFromExpression = Field(expr->getName());
    auto iter = std::find(allFields_.begin(), allFields_.end(), fieldFromExpression);
    if(iter == allFields_.end())
      allFields_.push_back(fieldFromExpression);
    this->ASTVisitorForwarding::Base::visit(expr);
  }

  const std::vector<Field>& getFields() const { return allFields_; }

private:
  std::vector<Field> allFields_;
};

extern std::vector<Field> getFieldFromStencilAST(const std::shared_ptr<AST>& ast) {
  FieldFinder finder;
  ast->accept(finder);
  return finder.getFields();
}

} // namespace sir
} // namespace dawn
