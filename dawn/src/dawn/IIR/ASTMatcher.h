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

#include "dawn/AST/ASTVisitor.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {
namespace iir {

//===------------------------------------------------------------------------------------------===//
//     ASTMatcher
//===------------------------------------------------------------------------------------------===//
/// @brief Traverses AST and return all matching statement or expression types
class ASTMatcher : public ast::ASTVisitorForwarding {
  iir::StencilInstantiation* instantiation_;
  iir::StencilMetaInformation& metadata_;
  ast::Stmt::Kind stmtKind_;
  ast::Expr::Kind exprKind_;
  std::vector<std::shared_ptr<ast::Stmt>> stmtMatches_;
  std::vector<std::shared_ptr<ast::Expr>> exprMatches_;

public:
  ASTMatcher(iir::StencilInstantiation* instantiation);

  std::vector<std::shared_ptr<ast::Stmt>>& match(ast::Stmt::Kind kind);
  std::vector<std::shared_ptr<ast::Expr>>& match(ast::Expr::Kind kind);

  void visit(const std::shared_ptr<iir::BlockStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::ExprStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::ReturnStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::IfStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) override;
  void visit(const std::shared_ptr<iir::UnaryOperator>& expr) override;
  void visit(const std::shared_ptr<iir::BinaryOperator>& expr) override;
  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override;
  void visit(const std::shared_ptr<iir::TernaryOperator>& expr) override;
  void visit(const std::shared_ptr<iir::FunCallExpr>& expr) override;
  void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override;
  void visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) override;
  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override;
  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override;
  void visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) override;

private:
  void iterate(iir::StencilInstantiation* instantiation);
  void iterate(const std::unique_ptr<iir::Stencil>& stencil);
  void iterate(const std::unique_ptr<iir::MultiStage>& multiStage);
  void iterate(const std::unique_ptr<iir::Stage>& stage);
  void iterate(const std::unique_ptr<iir::DoMethod>& doMethod);

  void check(const std::shared_ptr<ast::Stmt>& stmt);
  void check(const std::shared_ptr<ast::Expr>& expr);
};

} // namespace iir
} // namespace dawn
