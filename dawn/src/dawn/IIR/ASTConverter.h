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

#include "dawn/IIR/ASTStmt.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/AST/ASTVisitor.h"
#include <memory>
#include <unordered_map>

namespace dawn {

//===------------------------------------------------------------------------------------------===//
//     ASTConverter
//===------------------------------------------------------------------------------------------===//

/// @brief Converts an AST with SIR data to one (duplicated) with IIR data.
/// Can retrieve the converted nodes from the the stmt conversion map that is filled with the AST
/// visit.
class ASTConverter : public ast::ASTVisitorForwarding {
public:
  using StmtMap = std::unordered_map<std::shared_ptr<ast::Stmt>, std::shared_ptr<ast::Stmt>>;

  ASTConverter();

  StmtMap& getStmtMap();

  void visit(const std::shared_ptr<ast::BlockStmt>& blockStmt) override;
  void visit(const std::shared_ptr<ast::ExprStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::ReturnStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::VarDeclStmt>& varDeclStmt) override;
  void visit(const std::shared_ptr<ast::VerticalRegionDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::StencilCallDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::BoundaryConditionDeclStmt>& bcStmt) override;
  void visit(const std::shared_ptr<ast::IfStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::LoopStmt>& stmt) override;

private:
  StmtMap stmtMap_; // TODO: make it a pointer to first visited element
};

} // namespace dawn
