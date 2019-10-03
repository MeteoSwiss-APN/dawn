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

#ifndef DAWN_IIR_ASTSTMT_H
#define DAWN_IIR_ASTSTMT_H

#include "dawn/AST/ASTStmt.h"
#include <memory>
#include <optional>
#include <vector>

namespace dawn {
namespace iir {
/// @brief Data container for IIR stmt's data. All data should be optional and initially
/// uninitialized, meaning that such information hasn't yet been computed. The code that fills such
/// data should take care of its initialization.
struct IIRStmtData : public ast::StmtData {
  static const DataType ThisDataType = DataType::IIR_DATA_TYPE;

  bool operator==(const IIRStmtData&);
  bool operator!=(const IIRStmtData&);

  /// Stack trace of inlined stencil calls of this statement (might be empty).
  std::optional<std::vector<ast::StencilCall*>> StackTrace;

  DataType getDataType() const override { return ThisDataType; }
  std::unique_ptr<StmtData> clone() const override;
};

template <typename... Args>
std::shared_ptr<ast::BlockStmt> makeBlockStmt(Args&&... args) {
  return std::make_shared<ast::BlockStmt>(std::make_unique<IIRStmtData>(),
                                          std::forward<Args>(args)...);
}
template <typename... Args>
std::shared_ptr<ast::ExprStmt> makeExprStmt(Args&&... args) {
  return std::make_shared<ast::ExprStmt>(std::make_unique<IIRStmtData>(),
                                         std::forward<Args>(args)...);
}
template <typename... Args>
std::shared_ptr<ast::ReturnStmt> makeReturnStmt(Args&&... args) {
  return std::make_shared<ast::ReturnStmt>(std::make_unique<IIRStmtData>(),
                                           std::forward<Args>(args)...);
}
template <typename... Args>
std::shared_ptr<ast::VarDeclStmt> makeVarDeclStmt(Args&&... args) {
  return std::make_shared<ast::VarDeclStmt>(std::make_unique<IIRStmtData>(),
                                            std::forward<Args>(args)...);
}
template <typename... Args>
std::shared_ptr<ast::VerticalRegionDeclStmt> makeVerticalRegionDeclStmt(Args&&... args) {
  return std::make_shared<ast::VerticalRegionDeclStmt>(std::make_unique<IIRStmtData>(),
                                                       std::forward<Args>(args)...);
}
template <typename... Args>
std::shared_ptr<ast::StencilCallDeclStmt> makeStencilCallDeclStmt(Args&&... args) {
  return std::make_shared<ast::StencilCallDeclStmt>(std::make_unique<IIRStmtData>(),
                                                    std::forward<Args>(args)...);
}
template <typename... Args>
std::shared_ptr<ast::BoundaryConditionDeclStmt> makeBoundaryConditionDeclStmt(Args&&... args) {
  return std::make_shared<ast::BoundaryConditionDeclStmt>(std::make_unique<IIRStmtData>(),
                                                          std::forward<Args>(args)...);
}
template <typename... Args>
std::shared_ptr<ast::IfStmt> makeIfStmt(Args&&... args) {
  return std::make_shared<ast::IfStmt>(std::make_unique<IIRStmtData>(),
                                       std::forward<Args>(args)...);
}
//
// TODO refactor_AST: the following is going to be removed
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

} // namespace iir
} // namespace dawn

#endif
