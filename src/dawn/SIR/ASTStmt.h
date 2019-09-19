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

#ifndef DAWN_SIR_ASTSTMT_H
#define DAWN_SIR_ASTSTMT_H

#include "dawn/AST/ASTStmt.h"

namespace dawn {
namespace sir {
struct SIRStmtData : public ast::StmtData {
  static const DataType ThisDataType = DataType::SIR_DATA_TYPE;

  DataType getDataType() const override { return ThisDataType; }
  std::unique_ptr<StmtData> clone() const override;
};

template <typename... Args>
inline std::shared_ptr<ast::BlockStmt> makeBlockStmt(Args&&... args) {
  return std::move(std::make_shared<ast::BlockStmt>(new SIRStmtData(), args...));
}
template <typename... Args>
inline std::shared_ptr<ast::ExprStmt> makeExprStmt(Args&&... args) {
  return std::move(std::make_shared<ast::ExprStmt>(new SIRStmtData(), args...));
}
template <typename... Args>
inline std::shared_ptr<ast::ReturnStmt> makeReturnStmt(Args&&... args) {
  return std::move(std::make_shared<ast::ReturnStmt>(new SIRStmtData(), args...));
}
template <typename... Args>
inline std::shared_ptr<ast::VarDeclStmt> makeVarDeclStmt(Args&&... args) {
  return std::move(std::make_shared<ast::VarDeclStmt>(new SIRStmtData(), args...));
}
template <typename... Args>
inline std::shared_ptr<ast::VerticalRegionDeclStmt> makeVerticalRegionDeclStmt(Args&&... args) {
  return std::move(std::make_shared<ast::VerticalRegionDeclStmt>(new SIRStmtData(), args...));
}
template <typename... Args>
inline std::shared_ptr<ast::StencilCallDeclStmt> makeStencilCallDeclStmt(Args&&... args) {
  return std::move(std::make_shared<ast::StencilCallDeclStmt>(new SIRStmtData(), args...));
}
template <typename... Args>
inline std::shared_ptr<ast::BoundaryConditionDeclStmt>
makeBoundaryConditionDeclStmt(Args&&... args) {
  return std::move(std::make_shared<ast::BoundaryConditionDeclStmt>(new SIRStmtData(), args...));
}
template <typename... Args>
inline std::shared_ptr<ast::IfStmt> makeIfStmt(Args&&... args) {
  return std::move(std::make_shared<ast::IfStmt>(new SIRStmtData(), args...));
}
//
// TODO refactor_AST: this is TEMPORARY, will be removed in the future
//
using Stmt = ast::Stmt;
using BlockStmt = ast::BlockStmt;
using ExprStmt = ast::ExprStmt;
using ReturnStmt = ast::ReturnStmt;
using VarDeclStmt = ast::VarDeclStmt;
using VerticalRegionDeclStmt = ast::VerticalRegionDeclStmt;
using StencilCallDeclStmt = ast::StencilCallDeclStmt;
using BoundaryConditionDeclStmt = ast::BoundaryConditionDeclStmt;
using IfStmt = ast::IfStmt;

} // namespace sir
} // namespace dawn

#endif
