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

#ifndef DAWN_AST_ASTUTIL_H
#define DAWN_AST_ASTUTIL_H

#include "dawn/AST/ASTFwd.h"
#include <memory>
#include <unordered_map>

//  Need to have ASTHelper declared before including ASTVisitor.h, otherwise we end up with a
//  circular dependency.
namespace dawn {
namespace ast {
class ASTHelper {
public:
  template <typename Container, typename Type>
  static bool replaceOperands(std::shared_ptr<Type> const& oldExpr,
                              std::shared_ptr<Type> const& newExpr, Container& operands) {
    for(int i = 0; i < operands.size(); ++i) {
      if(operands[i] == oldExpr) {
        operands[i] = newExpr;
        return true;
      }
    }
    return false;
  }
};
} // namespace ast
} // namespace dawn

#include "dawn/AST/ASTVisitor.h"

namespace dawn {
namespace ast {

/// @brief Replace `oldExpr` with `newExpr` in `stmt`
///
/// Note that `oldExpr` is identfied by its `std::shared_ptr` and not its content. To replace
/// fields or variables in expr see `Optimizer/Replacing.cpp`.
///
/// @param stmt     Statement to analyze
/// @param oldExpr  Expression to replace
/// @param newExpr  Expression to use as a replacement
///
/// @ingroup ast
template <typename DataTraits>
extern void replaceOldExprWithNewExprInStmt(const std::shared_ptr<Stmt<DataTraits>>& stmt,
                                            const std::shared_ptr<Expr<DataTraits>>& oldExpr,
                                            const std::shared_ptr<Expr<DataTraits>>& newExpr);

/// @brief Visitor replace `oldExpr` with `newExpr` in the AST
template <typename DataTraits>
class ExprReplacer : virtual public ASTVisitor<DataTraits> {
  std::shared_ptr<Expr<DataTraits>> oldExpr_;
  std::shared_ptr<Expr<DataTraits>> newExpr_;

public:
  ExprReplacer(const std::shared_ptr<Expr<DataTraits>>& oldExpr,
               const std::shared_ptr<Expr<DataTraits>>& newExpr);

  void visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ExprStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ReturnStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<StencilCallDeclStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<UnaryOperator<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<BinaryOperator<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<AssignmentExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<TernaryOperator<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<FunCallExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<StencilFunCallExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<StencilFunArgExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<VarAccessExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<FieldAccessExpr<DataTraits>>& expr) override;
  void visit(const std::shared_ptr<LiteralAccessExpr<DataTraits>>& expr) override;
};

/// @brief Replace `oldStmt` with `newStmt` in `stmt`
///
/// Note that `oldStmt` is identfied by its `std::shared_ptr` and not its content.
///
/// @param stmt     Statement to analyze
/// @param oldStmt  Statement to replace
/// @param newStmt  Statement to use as a replacement
///
/// @ingroup ast
template <typename DataTraits>
extern void replaceOldStmtWithNewStmtInStmt(const std::shared_ptr<Stmt<DataTraits>>& stmt,
                                            const std::shared_ptr<Stmt<DataTraits>>& oldStmt,
                                            const std::shared_ptr<Stmt<DataTraits>>& newStmt);

/// @brief Visitor replacing `oldStmt` with `newStmt` in the AST
template <typename DataTraits>
class StmtReplacer : virtual public ASTVisitorForwarding<DataTraits> {
  std::shared_ptr<Stmt<DataTraits>> oldStmt_;
  std::shared_ptr<Stmt<DataTraits>> newStmt_;

public:
  StmtReplacer(const std::shared_ptr<Stmt<DataTraits>>& oldStmt,
               const std::shared_ptr<Stmt<DataTraits>>& newStmt);

  void visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) override;
};

/// @brief Try to evaluate the expression `expr`
///
/// Expressions can only be evaluated if they consist of unary, binary or ternary operators on
/// literals. Variable accesses (which reference literals or global constants) are only resolved if
/// a map of the value of each variable is provided. The value of these variable is currently
/// supplied as a `double` as this allows us to represent `int`s as well as `bool`s.
///
/// @param expr            Expression to evaluate
/// @param result          Result of the evaulation if successful, unmodified otherwise
/// @param variableMap     Map of variable name to respective value. If empty, no variables are
///                        resolved.
/// @returns `true` if evaluation was successful, `false` otherwise
///
/// @ingroup ast
/// @{
template <typename DataTraits>
extern bool evalExprAsDouble(const std::shared_ptr<Expr<DataTraits>>& expr, double& result,
                             const std::unordered_map<std::string, double>& variableMap =
                                 std::unordered_map<std::string, double>());
template <typename DataTraits>
extern bool evalExprAsInteger(const std::shared_ptr<Expr<DataTraits>>& expr, int& result,
                              const std::unordered_map<std::string, double>& variableMap =
                                  std::unordered_map<std::string, double>());
template <typename DataTraits>
extern bool evalExprAsBoolean(const std::shared_ptr<Expr<DataTraits>>& expr, bool& result,
                              const std::unordered_map<std::string, double>& variableMap =
                                  std::unordered_map<std::string, double>());
/// @}

} // namespace ast
} // namespace dawn

#include "dawn/AST/ASTUtil.tcc"

#endif
