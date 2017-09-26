//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_SIR_ASTASTVISITOR_H
#define GSL_SIR_ASTASTVISITOR_H

#include "gsl/SIR/ASTFwd.h"
#include <memory>

namespace gsl {

/// @brief Base class of all Visitor for ASTs and ASTNodes
/// @ingroup sir
class ASTVisitor {
public:
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

} // namespace gsl

#endif
