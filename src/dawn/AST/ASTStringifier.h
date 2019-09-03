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

#ifndef DAWN_AST_ASTSTRINGIFER_H
#define DAWN_AST_ASTSTRINGIFER_H

#include "dawn/AST/ASTFwd.h"
#include "dawn/AST/ASTVisitor.h"
#include <iosfwd>
#include <memory>
#include <sstream>
#include <string>

namespace dawn {
namespace ast {
/// @brief Dumps AST to string
///
/// (Used by ASTStringifier)
///
/// @ingroup ast
template <typename DataTraits>
class StringVisitor : virtual public ASTVisitor<DataTraits> {
protected:
  std::stringstream ss_;
  int curIndent_;
  int scopeDepth_;
  bool newLines_;

public:
  StringVisitor(int initialIndent, bool newLines);

  void visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ExprStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<ReturnStmt<DataTraits>>& stmt) override;
  void visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>>& stmt) override;
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

  inline std::string toString() const { return ss_.str(); }
};

/// @brief Pretty print the AST / ASTNodes using a C-like representation
///
/// This class is merely for debugging.
///
/// @ingroup ast
template <typename DataTraits, class Visitor = StringVisitor<DataTraits>>
struct ASTStringifier {
  ASTStringifier() = delete;

  static std::string toString(const std::shared_ptr<Stmt<DataTraits>>& stmt, int initialIndent = 0,
                              bool newLines = true);
  static std::string toString(const std::shared_ptr<Expr<DataTraits>>& expr, int initialIndent = 0,
                              bool newLines = true);
  static std::string toString(const AST<DataTraits>& ast, int initialIndent = 0,
                              bool newLines = true);
};

/// @name Stream operators
/// @ingroup ast
/// @{
template <typename DataTraits, class Visitor = StringVisitor<DataTraits>>
extern std::ostream& operator<<(std::ostream& os, const AST<DataTraits>& ast);
template <typename DataTraits, class Visitor = StringVisitor<DataTraits>>
extern std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Stmt<DataTraits>>& expr);
template <typename DataTraits, class Visitor = StringVisitor<DataTraits>>
extern std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Expr<DataTraits>>& stmt);
/// @}
} // namespace ast
} // namespace dawn

#include "dawn/AST/ASTStringifier.tcc"

#endif
