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

#ifndef DAWN_UNITTEST_PRINTALLEXPRESSINTYPES_H
#define DAWN_UNITTEST_PRINTALLEXPRESSINTYPES_H

#include "dawn/SIR/ASTExpr.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/ASTVisitor.h"

namespace dawn {
/// @brief Simple Visitor that prints Statements with their respective Types
/// @ingroup unittest
class PrintAllExpressionTypes : public ASTVisitorForwarding {
public:
  virtual void visit(const std::shared_ptr<BlockStmt>& stmt);
  virtual void visit(const std::shared_ptr<ExprStmt>& stmt);
  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt);
  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt);
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt);
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt);
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt);
  virtual void visit(const std::shared_ptr<IfStmt>& stmt);
  virtual void visit(const std::shared_ptr<UnaryOperator>& expr);
  virtual void visit(const std::shared_ptr<BinaryOperator>& expr);
  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr);
  virtual void visit(const std::shared_ptr<TernaryOperator>& expr);
  virtual void visit(const std::shared_ptr<FunCallExpr>& expr);
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr);
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr);
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr);
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr);
  virtual void visit(const std::shared_ptr<LiteralAccessExpr>& expr);
};

} // namespace dawn
#endif
