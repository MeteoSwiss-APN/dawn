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

#ifndef DAWN_SIR_ASTASTVISITOR_H
#define DAWN_SIR_ASTASTVISITOR_H

#include "dawn/SIR/ASTFwd.h"
#include <memory>

namespace dawn {

/// @brief Base class of all Visitor for ASTs and ASTNodes
/// @ingroup sir
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
  /// @}

  /// @brief Expressions
  /// @{
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
/// @ingroup sir
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
  /// @}

  /// @brief Expressions
  /// @{
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
/// @ingroup sir
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
  /// @}

  /// @brief Expressions
  /// @{
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
/// @ingroup sir
class ASTVisitorPostOrder {
public:
  virtual ~ASTVisitorPostOrder();

  /// @brief visitAndReplace will do a full traverse of this node for Statements
  /// @{
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<BlockStmt> stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<ExprStmt> stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<ReturnStmt> stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<VarDeclStmt> stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<VerticalRegionDeclStmt> stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<StencilCallDeclStmt> stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<BoundaryConditionDeclStmt> stmt);
  virtual std::shared_ptr<Stmt> visitAndReplace(std::shared_ptr<IfStmt> stmt);
  /// @}

  /// @brief visitAndReplace will do a full traverse of this node for Expressions
  /// @{
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<NOPExpr> expr) final;
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<UnaryOperator> expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<BinaryOperator> expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<AssignmentExpr> expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<TernaryOperator> expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<FunCallExpr> expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<StencilFunCallExpr> expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<StencilFunArgExpr> expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<VarAccessExpr> expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<FieldAccessExpr> expr);
  virtual std::shared_ptr<Expr> visitAndReplace(std::shared_ptr<LiteralAccessExpr> expr);

  /// @brief pre-visit the node for Statements and returns true if we should continue the tree
  /// traversal
  /// @{
  virtual bool preVisitNode(std::shared_ptr<BlockStmt> stmt);
  virtual bool preVisitNode(std::shared_ptr<ExprStmt> stmt);
  virtual bool preVisitNode(std::shared_ptr<ReturnStmt> stmt);
  virtual bool preVisitNode(std::shared_ptr<VarDeclStmt> stmt);
  virtual bool preVisitNode(std::shared_ptr<VerticalRegionDeclStmt> stmt);
  virtual bool preVisitNode(std::shared_ptr<StencilCallDeclStmt> stmt);
  virtual bool preVisitNode(std::shared_ptr<BoundaryConditionDeclStmt> stmt);
  virtual bool preVisitNode(std::shared_ptr<IfStmt> stmt);
  /// @}

  /// @brief pre-visit the node for Expressions and returns true if we should continue the tree
  /// traversal
  /// @{
  virtual bool preVisitNode(std::shared_ptr<NOPExpr> expr);
  virtual bool preVisitNode(std::shared_ptr<UnaryOperator> expr);
  virtual bool preVisitNode(std::shared_ptr<BinaryOperator> expr);
  virtual bool preVisitNode(std::shared_ptr<AssignmentExpr> expr);
  virtual bool preVisitNode(std::shared_ptr<TernaryOperator> expr);
  virtual bool preVisitNode(std::shared_ptr<FunCallExpr> expr);
  virtual bool preVisitNode(std::shared_ptr<StencilFunCallExpr> expr);
  virtual bool preVisitNode(std::shared_ptr<StencilFunArgExpr> expr);
  virtual bool preVisitNode(std::shared_ptr<VarAccessExpr> expr);
  virtual bool preVisitNode(std::shared_ptr<FieldAccessExpr> expr);
  virtual bool preVisitNode(std::shared_ptr<LiteralAccessExpr> expr);

  /// @}

  /// @brief post-visit the node for Statements and returns a modified/new version of the statement
  /// node to be returned to the parent
  /// @{
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<BlockStmt> stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<ExprStmt> stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<ReturnStmt> stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<VarDeclStmt> stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<VerticalRegionDeclStmt> stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<StencilCallDeclStmt> stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<BoundaryConditionDeclStmt> stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<IfStmt> stmt);
  /// @}

  /// @brief post-visit the node for Expressions and returns a modified/new version of the
  /// expression node to be returned to the parent
  /// @{
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<NOPExpr> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<UnaryOperator> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<BinaryOperator> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<AssignmentExpr> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<TernaryOperator> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<FunCallExpr> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<StencilFunCallExpr> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<StencilFunArgExpr> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<VarAccessExpr> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<FieldAccessExpr> expr);
  virtual std::shared_ptr<Expr> postVisitNode(std::shared_ptr<LiteralAccessExpr> expr);

  /// @}
};

/// @brief Visitor which forwards all calls to their children by default
/// @ingroup sir
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
  /// @}

  /// @brief Expressions
  /// @{
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
/// @ingroup sir
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
  /// @}

  /// @brief Expressions
  /// @{
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

} // namespace dawn

#endif
