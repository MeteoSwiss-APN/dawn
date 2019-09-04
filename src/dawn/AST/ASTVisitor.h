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

#ifndef DAWN_AST_ASTVISITOR_H
#define DAWN_AST_ASTVISITOR_H

#include "dawn/AST/ASTFwd.h"
#include <memory>

namespace dawn {
namespace ast {
/// @brief Base class of all Visitor for ASTs and ASTNodes
/// @ingroup ast
template <typename DataTraits>
class ASTVisitor {
public:
  virtual ~ASTVisitor();

  /// @brief Statements
  /// @{
  virtual void visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) = 0;
  virtual void visit(const std::shared_ptr<ExprStmt<DataTraits>>& stmt) = 0;
  virtual void visit(const std::shared_ptr<ReturnStmt<DataTraits>>& stmt) = 0;
  virtual void visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& stmt) = 0;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt<DataTraits>>& stmt) = 0;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>>& stmt) = 0;
  virtual void visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) = 0;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(const std::shared_ptr<NOPExpr<DataTraits>>& stmt) {}
  virtual void visit(const std::shared_ptr<UnaryOperator<DataTraits>>& expr) = 0;
  virtual void visit(const std::shared_ptr<BinaryOperator<DataTraits>>& expr) = 0;
  virtual void visit(const std::shared_ptr<AssignmentExpr<DataTraits>>& expr) = 0;
  virtual void visit(const std::shared_ptr<TernaryOperator<DataTraits>>& expr) = 0;
  virtual void visit(const std::shared_ptr<FunCallExpr<DataTraits>>& expr) = 0;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr<DataTraits>>& expr) = 0;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr<DataTraits>>& expr) = 0;
  virtual void visit(const std::shared_ptr<VarAccessExpr<DataTraits>>& expr) = 0;
  virtual void visit(const std::shared_ptr<FieldAccessExpr<DataTraits>>& expr) = 0;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr<DataTraits>>& expr) = 0;
  /// @}
};

/// @brief Base class of all Visitor for ASTs and ASTNodes
/// @ingroup ast
template <typename DataTraits>
class ASTVisitorNonConst {
public:
  virtual ~ASTVisitorNonConst();

  /// @brief Statements
  /// @{
  virtual void visit(std::shared_ptr<BlockStmt<DataTraits>> stmt) = 0;
  virtual void visit(std::shared_ptr<ExprStmt<DataTraits>> stmt) = 0;
  virtual void visit(std::shared_ptr<ReturnStmt<DataTraits>> stmt) = 0;
  virtual void visit(std::shared_ptr<VarDeclStmt<DataTraits>> stmt) = 0;
  virtual void visit(std::shared_ptr<StencilCallDeclStmt<DataTraits>> stmt) = 0;
  virtual void visit(std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>> stmt) = 0;
  virtual void visit(std::shared_ptr<IfStmt<DataTraits>> stmt) = 0;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(std::shared_ptr<NOPExpr<DataTraits>> expr) final {}
  virtual void visit(std::shared_ptr<UnaryOperator<DataTraits>> expr) = 0;
  virtual void visit(std::shared_ptr<BinaryOperator<DataTraits>> expr) = 0;
  virtual void visit(std::shared_ptr<AssignmentExpr<DataTraits>> expr) = 0;
  virtual void visit(std::shared_ptr<TernaryOperator<DataTraits>> expr) = 0;
  virtual void visit(std::shared_ptr<FunCallExpr<DataTraits>> expr) = 0;
  virtual void visit(std::shared_ptr<StencilFunCallExpr<DataTraits>> expr) = 0;
  virtual void visit(std::shared_ptr<StencilFunArgExpr<DataTraits>> expr) = 0;
  virtual void visit(std::shared_ptr<VarAccessExpr<DataTraits>> expr) = 0;
  virtual void visit(std::shared_ptr<FieldAccessExpr<DataTraits>> expr) = 0;
  virtual void visit(std::shared_ptr<LiteralAccessExpr<DataTraits>> expr) = 0;
  /// @}
};

/// @brief Visitor which forwards all calls to their children by default
/// @ingroup ast
template <typename DataTraits>
class ASTVisitorForwarding : public ASTVisitor<DataTraits> {
public:
  virtual ~ASTVisitorForwarding();

  using ASTVisitor<DataTraits>::visit;

  /// @brief Statements
  /// @{
  virtual void visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<ExprStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<ReturnStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) override;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(const std::shared_ptr<UnaryOperator<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<BinaryOperator<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<AssignmentExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<TernaryOperator<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<FunCallExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<VarAccessExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr<DataTraits>>& expr) override;
  /// @}
};

/// @brief Visitor post order that traverses the AST and replaces nodes according the post visit of
/// a node
/// Assume we have the following tree: A -> B, C. Then the traversal execution will do the
/// following:
///   previsit(A) (stop traversal if returns false)
///   B'=visitAndReplace(B) ->recursive
///   if(B' != B) A -> B'  (remove B)
///   C'=visitAndReplace(C) ->recursive
///   if(C' != C) A -> C' (remove B)
///   return postvisit(A) (modify parent A and return it to its parent)
/// @ingroup ast
template <typename DataTraits>
class ASTVisitorPostOrder {
public:
  virtual ~ASTVisitorPostOrder();

  /// @brief visitAndReplace will do a full traverse of this node for Statements
  /// @{
  virtual std::shared_ptr<Stmt<DataTraits>>
  visitAndReplace(std::shared_ptr<BlockStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  visitAndReplace(std::shared_ptr<ExprStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  visitAndReplace(std::shared_ptr<ReturnStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  visitAndReplace(std::shared_ptr<VarDeclStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  visitAndReplace(std::shared_ptr<StencilCallDeclStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  visitAndReplace(std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  visitAndReplace(std::shared_ptr<IfStmt<DataTraits>> const& stmt);
  /// @}

  /// @brief visitAndReplace will do a full traverse of this node for Expressions
  /// @{
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<NOPExpr<DataTraits>> const& expr) final;
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<UnaryOperator<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<BinaryOperator<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<AssignmentExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<TernaryOperator<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<FunCallExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<StencilFunCallExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<StencilFunArgExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<VarAccessExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<FieldAccessExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  visitAndReplace(std::shared_ptr<LiteralAccessExpr<DataTraits>> const& expr);

  /// @brief pre-visit the node for Statements and returns true if we should continue the tree
  /// traversal
  /// @{
  virtual bool preVisitNode(std::shared_ptr<BlockStmt<DataTraits>> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<ExprStmt<DataTraits>> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<ReturnStmt<DataTraits>> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<VarDeclStmt<DataTraits>> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<StencilCallDeclStmt<DataTraits>> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<IfStmt<DataTraits>> const& stmt);
  /// @}

  /// @brief pre-visit the node for Expressions and returns true if we should continue the tree
  /// traversal
  /// @{
  virtual bool preVisitNode(std::shared_ptr<NOPExpr<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<UnaryOperator<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<BinaryOperator<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<AssignmentExpr<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<TernaryOperator<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<FunCallExpr<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<StencilFunCallExpr<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<StencilFunArgExpr<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<VarAccessExpr<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<FieldAccessExpr<DataTraits>> const& expr);
  virtual bool preVisitNode(std::shared_ptr<LiteralAccessExpr<DataTraits>> const& expr);

  /// @}

  /// @brief post-visit the node for Statements and returns a modified/new version of the statement
  /// node to be returned to the parent
  /// @{
  virtual std::shared_ptr<Stmt<DataTraits>>
  postVisitNode(std::shared_ptr<BlockStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  postVisitNode(std::shared_ptr<ExprStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  postVisitNode(std::shared_ptr<ReturnStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  postVisitNode(std::shared_ptr<VarDeclStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  postVisitNode(std::shared_ptr<StencilCallDeclStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  postVisitNode(std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>> const& stmt);
  virtual std::shared_ptr<Stmt<DataTraits>>
  postVisitNode(std::shared_ptr<IfStmt<DataTraits>> const& stmt);
  /// @}

  /// @brief post-visit the node for Expressions and returns a modified/new version of the
  /// expression node to be returned to the parent
  /// @{
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<NOPExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<UnaryOperator<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<BinaryOperator<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<AssignmentExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<TernaryOperator<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<FunCallExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<StencilFunCallExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<StencilFunArgExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<VarAccessExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<FieldAccessExpr<DataTraits>> const& expr);
  virtual std::shared_ptr<Expr<DataTraits>>
  postVisitNode(std::shared_ptr<LiteralAccessExpr<DataTraits>> const& expr);

  /// @}
};

/// @brief Visitor which forwards all calls to their children by default
/// @ingroup ast
template <typename DataTraits>
class ASTVisitorForwardingNonConst : public ASTVisitorNonConst<DataTraits> {
public:
  virtual ~ASTVisitorForwardingNonConst();

  using ASTVisitorNonConst<DataTraits>::visit;

  /// @brief Statements
  /// @{
  virtual void visit(std::shared_ptr<BlockStmt<DataTraits>> stmt) override;
  virtual void visit(std::shared_ptr<ExprStmt<DataTraits>> stmt) override;
  virtual void visit(std::shared_ptr<ReturnStmt<DataTraits>> stmt) override;
  virtual void visit(std::shared_ptr<VarDeclStmt<DataTraits>> stmt) override;
  virtual void visit(std::shared_ptr<StencilCallDeclStmt<DataTraits>> stmt) override;
  virtual void visit(std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>> stmt) override;
  virtual void visit(std::shared_ptr<IfStmt<DataTraits>> stmt) override;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(std::shared_ptr<UnaryOperator<DataTraits>> expr) override;
  virtual void visit(std::shared_ptr<BinaryOperator<DataTraits>> expr) override;
  virtual void visit(std::shared_ptr<AssignmentExpr<DataTraits>> expr) override;
  virtual void visit(std::shared_ptr<TernaryOperator<DataTraits>> expr) override;
  virtual void visit(std::shared_ptr<FunCallExpr<DataTraits>> expr) override;
  virtual void visit(std::shared_ptr<StencilFunCallExpr<DataTraits>> expr) override;
  virtual void visit(std::shared_ptr<StencilFunArgExpr<DataTraits>> expr) override;
  virtual void visit(std::shared_ptr<VarAccessExpr<DataTraits>> expr) override;
  virtual void visit(std::shared_ptr<FieldAccessExpr<DataTraits>> expr) override;
  virtual void visit(std::shared_ptr<LiteralAccessExpr<DataTraits>> expr) override;
  /// @}
};

/// @brief Visitor which disables the visit of all expr and stmt
/// (in order to implement the corresponding functionality of a node, the method should be overrided
/// by the inherited class)
/// @ingroup ast
template <typename DataTraits>
class ASTVisitorDisabled : public ASTVisitor<DataTraits> {
public:
  virtual ~ASTVisitorDisabled();

  using ASTVisitor<DataTraits>::visit;

  /// @brief Statements
  /// @{
  virtual void visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<ExprStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<ReturnStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>>& stmt) override;
  virtual void visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) override;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(const std::shared_ptr<UnaryOperator<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<BinaryOperator<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<AssignmentExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<TernaryOperator<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<FunCallExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<VarAccessExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr<DataTraits>>& expr) override;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr<DataTraits>>& expr) override;
  /// @}
};

} // namespace ast
} // namespace dawn

#include "dawn/AST/ASTVisitor.tcc"

#endif
