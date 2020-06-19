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

#pragma once

#include "dawn/AST/ASTFwd.h"
#include <memory>

namespace dawn {
namespace ast {
/// @brief Base class of all Visitor for ASTs and ASTNodes
/// @ingroup ast
class ASTVisitor {
public:
  virtual ~ASTVisitor();

  /// @brief Statements
  /// @{
  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) = 0;
  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) = 0;
  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) = 0;
  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) = 0;
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) = 0;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) = 0;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) = 0;
  virtual void visit(const std::shared_ptr<IfStmt>& stmt) = 0;
  virtual void visit(const std::shared_ptr<LoopStmt>& stmt) = 0;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(const std::shared_ptr<ReductionOverNeighborExpr>& expr) = 0;
  virtual void visit(const std::shared_ptr<NOPExpr>& stmt) final {}
  virtual void visit(const std::shared_ptr<UnaryOperator>& expr) = 0;
  virtual void visit(const std::shared_ptr<BinaryOperator>& expr) = 0;
  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr) = 0;
  virtual void visit(const std::shared_ptr<TernaryOperator>& expr) = 0;
  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) = 0;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) = 0;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) = 0;
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) = 0;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) = 0;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr>& expr) = 0;
  /// @}
};

/// @brief Base class of all Visitor for ASTs and ASTNodes
/// @ingroup ast
class ASTVisitorNonConst {
public:
  virtual ~ASTVisitorNonConst();

  /// @brief Statements
  /// @{
  virtual void visit(std::shared_ptr<BlockStmt> stmt) = 0;
  virtual void visit(std::shared_ptr<ExprStmt> stmt) = 0;
  virtual void visit(std::shared_ptr<ReturnStmt> stmt) = 0;
  virtual void visit(std::shared_ptr<VarDeclStmt> stmt) = 0;
  virtual void visit(std::shared_ptr<VerticalRegionDeclStmt> stmt) = 0;
  virtual void visit(std::shared_ptr<StencilCallDeclStmt> stmt) = 0;
  virtual void visit(std::shared_ptr<BoundaryConditionDeclStmt> stmt) = 0;
  virtual void visit(std::shared_ptr<IfStmt> stmt) = 0;
  virtual void visit(std::shared_ptr<LoopStmt> stmt) = 0;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(std::shared_ptr<ReductionOverNeighborExpr> expr) = 0;
  virtual void visit(std::shared_ptr<NOPExpr> expr) final {}
  virtual void visit(std::shared_ptr<UnaryOperator> expr) = 0;
  virtual void visit(std::shared_ptr<BinaryOperator> expr) = 0;
  virtual void visit(std::shared_ptr<AssignmentExpr> expr) = 0;
  virtual void visit(std::shared_ptr<TernaryOperator> expr) = 0;
  virtual void visit(std::shared_ptr<FunCallExpr> expr) = 0;
  virtual void visit(std::shared_ptr<StencilFunCallExpr> expr) = 0;
  virtual void visit(std::shared_ptr<StencilFunArgExpr> expr) = 0;
  virtual void visit(std::shared_ptr<VarAccessExpr> expr) = 0;
  virtual void visit(std::shared_ptr<FieldAccessExpr> expr) = 0;
  virtual void visit(std::shared_ptr<LiteralAccessExpr> expr) = 0;
  /// @}
};

/// @brief Visitor which forwards all calls to their children by default
/// @ingroup ast
class ASTVisitorForwarding : public ASTVisitor {
public:
  virtual ~ASTVisitorForwarding();

  /// @brief Statements
  /// @{
  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<IfStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<LoopStmt>& stmt) override;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(const std::shared_ptr<ReductionOverNeighborExpr>& expr) override;
  virtual void visit(const std::shared_ptr<UnaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<BinaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr) override;
  virtual void visit(const std::shared_ptr<TernaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override;
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
class ASTVisitorPostOrder {
public:
  virtual ~ASTVisitorPostOrder();

  /// @brief visitAndReplace will do a full traverse of this node for Statements
  /// @{
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<BlockStmt> const& stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<ExprStmt> const& stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<ReturnStmt> const& stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<VarDeclStmt> const& stmt);
  virtual std::shared_ptr<Stmt>
  visitAndReplace(std::shared_ptr<VerticalRegionDeclStmt> const& stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<StencilCallDeclStmt> const& stmt);
  virtual std::shared_ptr<Stmt>
  visitAndReplace(std::shared_ptr<BoundaryConditionDeclStmt> const& stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<IfStmt> const& stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<LoopStmt> const& stmt);
  /// @}

  /// @brief visitAndReplace will do a full traverse of this node for Expressions
  /// @{
  virtual std::shared_ptr<Expr>
  visitAndReplace(std::shared_ptr<ReductionOverNeighborExpr> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<NOPExpr> const& expr) final;
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<UnaryOperator> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<BinaryOperator> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<AssignmentExpr> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<TernaryOperator> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<FunCallExpr> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<StencilFunCallExpr> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<StencilFunArgExpr> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<VarAccessExpr> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<FieldAccessExpr> const& expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<LiteralAccessExpr> const& expr);

  /// @brief pre-visit the node for Statements and returns true if we should continue the tree
  /// traversal
  /// @{
  virtual bool preVisitNode(std::shared_ptr<BlockStmt> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<ExprStmt> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<ReturnStmt> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<VarDeclStmt> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<VerticalRegionDeclStmt> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<StencilCallDeclStmt> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<BoundaryConditionDeclStmt> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<IfStmt> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<LoopStmt> const& stmt);
  /// @}

  /// @brief pre-visit the node for Expressions and returns true if we should continue the tree
  /// traversal
  /// @{
  virtual bool preVisitNode(std::shared_ptr<ReductionOverNeighborExpr> const& expr);
  virtual bool preVisitNode(std::shared_ptr<NOPExpr> const& expr);
  virtual bool preVisitNode(std::shared_ptr<UnaryOperator> const& expr);
  virtual bool preVisitNode(std::shared_ptr<BinaryOperator> const& expr);
  virtual bool preVisitNode(std::shared_ptr<AssignmentExpr> const& expr);
  virtual bool preVisitNode(std::shared_ptr<TernaryOperator> const& expr);
  virtual bool preVisitNode(std::shared_ptr<FunCallExpr> const& expr);
  virtual bool preVisitNode(std::shared_ptr<StencilFunCallExpr> const& expr);
  virtual bool preVisitNode(std::shared_ptr<StencilFunArgExpr> const& expr);
  virtual bool preVisitNode(std::shared_ptr<VarAccessExpr> const& expr);
  virtual bool preVisitNode(std::shared_ptr<FieldAccessExpr> const& expr);
  virtual bool preVisitNode(std::shared_ptr<LiteralAccessExpr> const& expr);

  /// @}

  /// @brief post-visit the node for Statements and returns a modified/new version of the statement
  /// node to be returned to the parent
  /// @{
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<BlockStmt> const& stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<ExprStmt> const& stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<ReturnStmt> const& stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<VarDeclStmt> const& stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<VerticalRegionDeclStmt> const& stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<StencilCallDeclStmt> const& stmt);
  virtual std::shared_ptr<Stmt>
  postVisitNode(std::shared_ptr<BoundaryConditionDeclStmt> const& stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<IfStmt> const& stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<LoopStmt> const& stmt);
  /// @}

  /// @brief post-visit the node for Expressions and returns a modified/new version of the
  /// expression node to be returned to the parent
  /// @{
  virtual std::shared_ptr<Expr>
  postVisitNode(std::shared_ptr<ReductionOverNeighborExpr> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<NOPExpr> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<UnaryOperator> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<BinaryOperator> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<AssignmentExpr> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<TernaryOperator> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<FunCallExpr> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<StencilFunCallExpr> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<StencilFunArgExpr> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<VarAccessExpr> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<FieldAccessExpr> const& expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<LiteralAccessExpr> const& expr);

  /// @}
};

/// @brief Visitor which forwards all calls to their children by default
/// @ingroup ast
class ASTVisitorForwardingNonConst : public ASTVisitorNonConst {
public:
  virtual ~ASTVisitorForwardingNonConst();

  /// @brief Statements
  /// @{
  virtual void visit(std::shared_ptr<BlockStmt> stmt) override;
  virtual void visit(std::shared_ptr<ExprStmt> stmt) override;
  virtual void visit(std::shared_ptr<ReturnStmt> stmt) override;
  virtual void visit(std::shared_ptr<VarDeclStmt> stmt) override;
  virtual void visit(std::shared_ptr<VerticalRegionDeclStmt> stmt) override;
  virtual void visit(std::shared_ptr<StencilCallDeclStmt> stmt) override;
  virtual void visit(std::shared_ptr<BoundaryConditionDeclStmt> stmt) override;
  virtual void visit(std::shared_ptr<IfStmt> stmt) override;
  virtual void visit(std::shared_ptr<LoopStmt> stmt) override;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(std::shared_ptr<ReductionOverNeighborExpr> expr) override;
  virtual void visit(std::shared_ptr<UnaryOperator> expr) override;
  virtual void visit(std::shared_ptr<BinaryOperator> expr) override;
  virtual void visit(std::shared_ptr<AssignmentExpr> expr) override;
  virtual void visit(std::shared_ptr<TernaryOperator> expr) override;
  virtual void visit(std::shared_ptr<FunCallExpr> expr) override;
  virtual void visit(std::shared_ptr<StencilFunCallExpr> expr) override;
  virtual void visit(std::shared_ptr<StencilFunArgExpr> expr) override;
  virtual void visit(std::shared_ptr<VarAccessExpr> expr) override;
  virtual void visit(std::shared_ptr<FieldAccessExpr> expr) override;
  virtual void visit(std::shared_ptr<LiteralAccessExpr> expr) override;
  /// @}
};

/// @brief Visitor which disables the visit of all expr and stmt
/// (in order to implement the corresponding functionality of a node, the method should be overrided
/// by the inherited class)
/// @ingroup ast
class ASTVisitorDisabled : public ASTVisitor {
public:
  virtual ~ASTVisitorDisabled();

  /// @brief Statements
  /// @{
  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<IfStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<LoopStmt>& stmt) override;
  /// @}

  /// @brief Expressions
  /// @{
  virtual void visit(const std::shared_ptr<ReductionOverNeighborExpr>& expr) override;
  virtual void visit(const std::shared_ptr<UnaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<BinaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr) override;
  virtual void visit(const std::shared_ptr<TernaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override;
  /// @}
};

} // namespace ast
} // namespace dawn
