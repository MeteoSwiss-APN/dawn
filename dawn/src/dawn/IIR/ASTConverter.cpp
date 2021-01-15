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

#include "dawn/IIR/ASTConverter.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Unreachable.h"
#include <memory>
#include <unordered_map>

namespace dawn {

ASTConverter::ASTConverter() {}

ASTConverter::StmtMap& ASTConverter::getStmtMap() {
  // TODO: this class whould need to be tested. Here is a possible test:
  //  for(const std::pair<std::shared_ptr<sir::Stmt>, std::shared_ptr<iir::Stmt>>& pair : stmtMap_)
  //  {
  //      DAWN_ASSERT((int)pair.first->getKind() == (int)pair.second->getKind());
  //  }
  return stmtMap_;
}

void ASTConverter::visit(const std::shared_ptr<ast::BlockStmt>& blockStmt) {
  ast::BlockStmt::StatementList statementList;
  for(const auto& stmt : blockStmt->getStatements()) {
    stmt->accept(*this);
    statementList.push_back(stmtMap_.at(stmt));
  }
  stmtMap_.emplace(blockStmt, iir::makeBlockStmt(statementList, blockStmt->getSourceLocation()));
}

void ASTConverter::visit(const std::shared_ptr<ast::ExprStmt>& stmt) {
  stmtMap_.emplace(stmt, iir::makeExprStmt(stmt->getExpr()->clone(), stmt->getSourceLocation()));
}

void ASTConverter::visit(const std::shared_ptr<ast::ReturnStmt>& stmt) {
  stmtMap_.emplace(stmt, iir::makeReturnStmt(stmt->getExpr()->clone(), stmt->getSourceLocation()));
}

void ASTConverter::visit(const std::shared_ptr<ast::VarDeclStmt>& varDeclStmt) {
  ast::VarDeclStmt::InitList initList;
  for(auto& expr : varDeclStmt->getInitList())
    initList.push_back(expr->clone());

  stmtMap_.emplace(varDeclStmt,
                   iir::makeVarDeclStmt(varDeclStmt->getType(), varDeclStmt->getName(),
                                        varDeclStmt->getDimension(), varDeclStmt->getOp(), initList,
                                        varDeclStmt->getSourceLocation()));
}

void ASTConverter::visit(const std::shared_ptr<ast::VerticalRegionDeclStmt>& stmt) {
  stmt->getVerticalRegion()->Ast->getRoot()->accept(*this);

  auto verticalRegion = std::make_shared<sir::VerticalRegion>(
      std::make_shared<ast::AST>(std::dynamic_pointer_cast<ast::BlockStmt>(
          stmtMap_.at(stmt->getVerticalRegion()->Ast->getRoot()))),
      stmt->getVerticalRegion()->VerticalInterval, stmt->getVerticalRegion()->LoopOrder,
      stmt->getVerticalRegion()->IterationSpace[0], stmt->getVerticalRegion()->IterationSpace[1],
      stmt->getVerticalRegion()->Loc);
  verticalRegion->IterationSpace = stmt->getVerticalRegion()->IterationSpace;

  stmtMap_.emplace(stmt,
                   iir::makeVerticalRegionDeclStmt(verticalRegion, stmt->getSourceLocation()));
}

void ASTConverter::visit(const std::shared_ptr<ast::StencilCallDeclStmt>& stmt) {
  stmtMap_.emplace(stmt, iir::makeStencilCallDeclStmt(stmt->getStencilCall()->clone(),
                                                      stmt->getSourceLocation()));
}

void ASTConverter::visit(const std::shared_ptr<ast::BoundaryConditionDeclStmt>& bcStmt) {
  std::shared_ptr<ast::BoundaryConditionDeclStmt> iirBcStmt =
      iir::makeBoundaryConditionDeclStmt(bcStmt->getFunctor(), bcStmt->getSourceLocation());
  iirBcStmt->getFields() = bcStmt->getFields();
  stmtMap_.emplace(bcStmt, iirBcStmt);
}

void ASTConverter::visit(const std::shared_ptr<ast::IfStmt>& stmt) {
  stmt->getCondStmt()->accept(*this);
  stmt->getThenStmt()->accept(*this);
  if(stmt->hasElse())
    stmt->getElseStmt()->accept(*this);
  stmtMap_.emplace(
      stmt, iir::makeIfStmt(stmtMap_.at(stmt->getCondStmt()), stmtMap_.at(stmt->getThenStmt()),
                            stmt->hasElse() ? stmtMap_.at(stmt->getElseStmt()) : nullptr,
                            stmt->getSourceLocation()));
}

void ASTConverter::visit(const std::shared_ptr<ast::LoopStmt>& stmt) {
  stmt->getBlockStmt()->accept(*this);
  if(auto* chainDesc =
         dynamic_cast<const ast::ChainIterationDescr*>(stmt->getIterationDescrPtr())) {
    stmtMap_.emplace(stmt, iir::makeLoopStmt(chainDesc->getChain(), chainDesc->getIncludeCenter(),
                                             std::dynamic_pointer_cast<ast::BlockStmt>(
                                                 stmtMap_.at(stmt->getBlockStmt()))));
  } else {
    dawn_unreachable("unsupported loop descriptor!\n");
  }
}

} // namespace dawn
