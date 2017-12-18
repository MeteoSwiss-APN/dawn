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
