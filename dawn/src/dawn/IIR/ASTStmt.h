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

#include "dawn/AST/ASTStmt.h"
#include "dawn/IIR/Accesses.h"
#include <memory>
#include <optional>
#include <vector>

namespace dawn {
namespace iir {

class Extents;

/// @brief Data container for IIR stmt's data. All data should be optional and initially
/// uninitialized, meaning that such information hasn't yet been computed. The code that fills such
/// data should take care of its initialization.
struct IIRStmtData : public ast::StmtData {
  static const DataType ThisDataType = DataType::IIR_DATA_TYPE;

  bool operator==(const IIRStmtData&) const;
  bool operator!=(const IIRStmtData&) const;

  /// Stack trace of inlined stencil calls of this statement (might be empty).
  std::optional<std::vector<ast::StencilCall*>> StackTrace;

  /// In case of a non function call stmt, the accesses are stored in CallerAccesses, while
  /// CalleeAccesses is uninitialized

  /// Accesses of the statement. If the statement is part of a stencil-function, this will store the
  /// caller accesses. The caller access will have the initial offset added (e.g if a stencil
  /// function is called with `avg(u(i+1))` the initial offset of `u` is `[1, 0, 0]`).
  std::optional<Accesses> CallerAccesses;

  /// If the statement is part of a stencil-function, this will store the callee accesses i.e the
  /// accesses without the initial offset of the call
  std::optional<Accesses> CalleeAccesses;

  DataType getDataType() const override { return ThisDataType; }
  std::unique_ptr<StmtData> clone() const override;
  bool equals(StmtData const* other) const override;

  std::string toString(std::function<std::string(int)>&& accessIDToStringFunction,
                       std::size_t initialIndent = 0) const;
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

struct VarDeclStmtData : public IIRStmtData {

  VarDeclStmtData() = default;
  VarDeclStmtData(const VarDeclStmtData&);

  bool operator==(const VarDeclStmtData&) const;
  bool operator!=(const VarDeclStmtData&) const;

  /// ID of the variable declared by the statement
  std::optional<int> AccessID;

  std::unique_ptr<StmtData> clone() const override;
  bool equals(StmtData const* other) const override;
};

template <typename... Args>
std::shared_ptr<ast::VarDeclStmt> makeVarDeclStmt(Args&&... args) {
  return std::make_shared<ast::VarDeclStmt>(std::make_unique<VarDeclStmtData>(),
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
template <typename... Args>
std::shared_ptr<ast::LoopStmt> makeLoopStmt(Args&&... args) {
  return std::make_shared<ast::LoopStmt>(std::make_unique<IIRStmtData>(),
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
using LoopStmt = ast::LoopStmt;
//
// END_TODO
//
/// @brief Computes the maximum extent among all the accesses of accessID in stmt
std::optional<Extents> computeMaximumExtents(Stmt& stmt, const int accessID);

/// @brief Get the `AccessID` of the a VarDeclStmt
int getAccessID(const std::shared_ptr<VarDeclStmt>& stmt);

} // namespace iir
} // namespace dawn
