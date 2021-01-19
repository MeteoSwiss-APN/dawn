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
#include "dawn/SIR/ASTVisitor.h"
#include <memory>
#include <unordered_map>

namespace dawn {

//===------------------------------------------------------------------------------------------===//
//     ASTConverter
//===------------------------------------------------------------------------------------------===//

/// @brief Converts an AST with SIR data to one (duplicated) with IIR data.
/// Can retrieve the converted nodes from the the stmt conversion map that is filled with the AST
/// visit.
class ASTConverter : public sir::ASTVisitorForwarding {
public:
  using StmtMap = std::unordered_map<std::shared_ptr<sir::Stmt>, std::shared_ptr<iir::Stmt>>;

  ASTConverter();

  StmtMap& getStmtMap();

  void visit(const std::shared_ptr<sir::BlockStmt>& blockStmt) override;
  void visit(const std::shared_ptr<sir::ExprStmt>& stmt) override;
  void visit(const std::shared_ptr<sir::ReturnStmt>& stmt) override;
  void visit(const std::shared_ptr<sir::VarDeclStmt>& varDeclStmt) override;
  void visit(const std::shared_ptr<sir::VerticalRegionDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<sir::StencilCallDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<sir::BoundaryConditionDeclStmt>& bcStmt) override;
  void visit(const std::shared_ptr<sir::IfStmt>& stmt) override;
  void visit(const std::shared_ptr<sir::LoopStmt>& stmt) override;

private:
  StmtMap stmtMap_; // TODO: make it a pointer to first visited element
};

} // namespace dawn
