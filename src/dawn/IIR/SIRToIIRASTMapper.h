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

#ifndef DAWN_IIR_SIRTOIIRASTMAPPER_H
#define DAWN_IIR_SIRTOIIRASTMAPPER_H

#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/SIR/ASTExpr.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/ASTVisitor.h"
#include <memory>
#include <unordered_map>

namespace dawn {

//===------------------------------------------------------------------------------------------===//
//     SIRToIIRASTMapper
//===------------------------------------------------------------------------------------------===//

/// @brief Produces maps from sir Stmts/Exprs to their conversion into iir Stmts/Exprs.
/// sir::VerticalRegionDeclStmt is mapped to its underlying BlockStmt.
class SIRToIIRASTMapper : public sir::ASTVisitor {

public:
  using StmtMap = std::unordered_map<std::shared_ptr<sir::Stmt>, std::shared_ptr<iir::Stmt>>;
  using ExprMap = std::unordered_map<std::shared_ptr<sir::Expr>, std::shared_ptr<iir::Expr>>;

  SIRToIIRASTMapper() {}
  /// @brief Statements
  /// @{
  void visit(const std::shared_ptr<sir::VerticalRegionDeclStmt>& stmt) override {
    stmt->getAST()->getRoot()->accept(*this);
    stmtMap_.emplace(stmt, stmtMap_.at(stmt->getAST()->getRoot()));
  }
  void visit(const std::shared_ptr<sir::BlockStmt>& blockStmt) override {
    iir::BlockStmt::StatementList statementList;
    for(auto& stmt : blockStmt->getStatements()) {
      stmt->accept(*this);
      statementList.push_back(stmtMap_.at(stmt));
    }
    stmtMap_.emplace(
        blockStmt, std::make_shared<iir::BlockStmt>(statementList, blockStmt->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::ExprStmt>& stmt) override {
    stmt->getExpr()->accept(*this);
    stmtMap_.emplace(stmt, std::make_shared<iir::ExprStmt>(exprMap_.at(stmt->getExpr()),
                                                           stmt->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::ReturnStmt>& stmt) override {
    stmt->getExpr()->accept(*this);
    stmtMap_.emplace(stmt, std::make_shared<iir::ReturnStmt>(exprMap_.at(stmt->getExpr()),
                                                             stmt->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::VarDeclStmt>& varDeclStmt) override {
    iir::VarDeclStmt::InitList initList;
    for(auto& expr : varDeclStmt->getInitList()) {
      expr->accept(*this);
      initList.push_back(exprMap_.at(expr));
    }
    stmtMap_.emplace(varDeclStmt, std::make_shared<iir::VarDeclStmt>(
                                      varDeclStmt->getType(), varDeclStmt->getName(),
                                      varDeclStmt->getDimension(), varDeclStmt->getOp(), initList,
                                      varDeclStmt->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::StencilCallDeclStmt>& stmt) override {
    stmtMap_.emplace(stmt, std::make_shared<iir::StencilCallDeclStmt>(
                               stmt->getStencilCall()->clone(), stmt->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::BoundaryConditionDeclStmt>& bcStmt) override {
    std::shared_ptr<iir::BoundaryConditionDeclStmt> iirBcStmt =
        std::make_shared<iir::BoundaryConditionDeclStmt>(bcStmt->getFunctor(),
                                                         bcStmt->getSourceLocation());
    iirBcStmt->getFields() = bcStmt->getFields();
    stmtMap_.emplace(bcStmt, iirBcStmt);
  }
  void visit(const std::shared_ptr<sir::IfStmt>& stmt) override {
    stmt->getCondStmt()->accept(*this);
    stmt->getThenStmt()->accept(*this);
    if(stmt->hasElse())
      stmt->getElseStmt()->accept(*this);
    stmtMap_.emplace(stmt, std::make_shared<iir::IfStmt>(
                               stmtMap_.at(stmt->getCondStmt()), stmtMap_.at(stmt->getThenStmt()),
                               stmt->hasElse() ? stmtMap_.at(stmt->getElseStmt()) : nullptr,
                               stmt->getSourceLocation()));
  }
  /// @}
  /// @brief Expressions
  /// @{
  void visit(const std::shared_ptr<sir::NOPExpr>& expr) override {
    exprMap_.emplace(expr, std::make_shared<iir::NOPExpr>(expr->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::UnaryOperator>& expr) override {
    expr->getOperand()->accept(*this);
    exprMap_.emplace(expr, std::make_shared<iir::UnaryOperator>(exprMap_.at(expr->getOperand()),
                                                                expr->getOp(),
                                                                expr->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::BinaryOperator>& expr) override {
    expr->getLeft()->accept(*this);
    expr->getRight()->accept(*this);
    exprMap_.emplace(expr, std::make_shared<iir::BinaryOperator>(
                               exprMap_.at(expr->getLeft()), expr->getOp(),
                               exprMap_.at(expr->getRight()), expr->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::AssignmentExpr>& expr) override {
    expr->getLeft()->accept(*this);
    expr->getRight()->accept(*this);
    exprMap_.emplace(expr, std::make_shared<iir::AssignmentExpr>(
                               exprMap_.at(expr->getLeft()), exprMap_.at(expr->getRight()),
                               expr->getOp(), expr->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::TernaryOperator>& expr) override {
    expr->getCondition()->accept(*this);
    expr->getLeft()->accept(*this);
    expr->getRight()->accept(*this);
    exprMap_.emplace(expr, std::make_shared<iir::TernaryOperator>(
                               exprMap_.at(expr->getCondition()), exprMap_.at(expr->getLeft()),
                               exprMap_.at(expr->getRight()), expr->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::FunCallExpr>& funCallExpr) override {
    iir::FunCallExpr::ArgumentsList argList;
    for(auto& arg : funCallExpr->getArguments()) {
      arg->accept(*this);
      argList.push_back(exprMap_.at(arg));
    }
    auto iirFunCallExpr = std::make_shared<iir::FunCallExpr>(funCallExpr->getCallee(),
                                                             funCallExpr->getSourceLocation());
    iirFunCallExpr->getArguments() = argList;
    exprMap_.emplace(funCallExpr, iirFunCallExpr);
  }
  void visit(const std::shared_ptr<sir::StencilFunCallExpr>& funCallExpr) override {
    iir::StencilFunCallExpr::ArgumentsList argList;
    for(auto& arg : funCallExpr->getArguments()) {
      arg->accept(*this);
      argList.push_back(exprMap_.at(arg));
    }
    auto iirFunCallExpr = std::make_shared<iir::StencilFunCallExpr>(
        funCallExpr->getCallee(), funCallExpr->getSourceLocation());
    iirFunCallExpr->getArguments() = argList;
    exprMap_.emplace(funCallExpr, iirFunCallExpr);
  }
  void visit(const std::shared_ptr<sir::StencilFunArgExpr>& expr) override {
    exprMap_.emplace(expr, std::make_shared<iir::StencilFunArgExpr>(
                               expr->getDimension(), expr->getOffset(), expr->getArgumentIndex(),
                               expr->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::VarAccessExpr>& expr) override {
    if(expr->isArrayAccess())
      expr->getIndex()->accept(*this);
    auto iirExpr = std::make_shared<iir::VarAccessExpr>(
        expr->getName(), expr->isArrayAccess() ? exprMap_.at(expr->getIndex()) : nullptr,
        expr->getSourceLocation());
    iirExpr->setIsExternal(expr->isExternal());
    exprMap_.emplace(expr, iirExpr);
  }
  void visit(const std::shared_ptr<sir::FieldAccessExpr>& expr) override {
    exprMap_.emplace(expr, std::make_shared<iir::FieldAccessExpr>(
                               expr->getName(), expr->getOffset(), expr->getArgumentMap(),
                               expr->getArgumentOffset(), expr->negateOffset(),
                               expr->getSourceLocation()));
  }
  void visit(const std::shared_ptr<sir::LiteralAccessExpr>& expr) override {
    exprMap_.emplace(expr, std::make_shared<iir::LiteralAccessExpr>(expr->getValue(),
                                                                    expr->getBuiltinType(),
                                                                    expr->getSourceLocation()));
  }
  /// @}

  inline StmtMap& getStmtMap() {
    for(const std::pair<std::shared_ptr<sir::Stmt>, std::shared_ptr<iir::Stmt>>& pair : stmtMap_) {
      if(pair.first->getKind() == sir::Stmt::StmtKind::SK_VerticalRegionDeclStmt)
        DAWN_ASSERT(pair.second->getKind() == iir::Stmt::StmtKind::SK_BlockStmt);
      else
        DAWN_ASSERT((int)pair.first->getKind() == (int)pair.second->getKind());
    }
    return stmtMap_;
  }
  inline ExprMap& getExprMap() {
    for(const std::pair<std::shared_ptr<sir::Expr>, std::shared_ptr<iir::Expr>>& pair : exprMap_) {
      DAWN_ASSERT((int)pair.first->getKind() == (int)pair.second->getKind());
    }
    return exprMap_;
  }

private:
  StmtMap stmtMap_;
  ExprMap exprMap_;
};

} // namespace dawn

#endif
