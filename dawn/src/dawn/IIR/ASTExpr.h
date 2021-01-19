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

/// @brief Get the `AccessID` of the Expr (VarAccess or FieldAccess or LiteralAccess)
int getAccessID(const std::shared_ptr<ast::Expr>& expr);

} // namespace iir
} // namespace dawn
