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

#include "dawn/AST/ASTExpr.h"

#include <optional>

namespace dawn {
namespace iir {

struct IIRAccessExprData : public ast::AccessExprData {
  IIRAccessExprData() = default;
  IIRAccessExprData(const IIRAccessExprData& other);

  bool operator==(const IIRAccessExprData&) const;
  bool operator!=(const IIRAccessExprData&) const;

  std::unique_ptr<ast::AccessExprData> clone() const override;

  bool equals(AccessExprData const* other) const override;

  /// ID of the resource (literal or variable or field) accessed by the expression
  std::optional<int> AccessID;
};

//
// TODO refactor_AST: this is TEMPORARY, will be removed in the future
//
using Expr = ast::Expr;
using NOPExpr = ast::NOPExpr;
using UnaryOperator = ast::UnaryOperator;
using BinaryOperator = ast::BinaryOperator;
using AssignmentExpr = ast::AssignmentExpr;
using TernaryOperator = ast::TernaryOperator;
using FunCallExpr = ast::FunCallExpr;
using StencilFunCallExpr = ast::StencilFunCallExpr;
using StencilFunArgExpr = ast::StencilFunArgExpr;
using VarAccessExpr = ast::VarAccessExpr;
using FieldAccessExpr = ast::FieldAccessExpr;
using LiteralAccessExpr = ast::LiteralAccessExpr;
using ReductionOverNeighborExpr = ast::ReductionOverNeighborExpr;
//
// END_TODO
//

/// @brief Get the `AccessID` of the Expr (VarAccess or FieldAccess or LiteralAccess)
int getAccessID(const std::shared_ptr<Expr>& expr);

} // namespace iir
} // namespace dawn
