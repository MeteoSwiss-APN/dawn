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

#ifndef DAWN_IIR_ASTEXPR_H
#define DAWN_IIR_ASTEXPR_H

#include "dawn/AST/ASTExpr.h"
#include <boost/optional.hpp>

namespace dawn {
namespace iir {

struct IIRVarAccessExprData : public ast::VarAccessExprData {
  IIRVarAccessExprData() = default;
  IIRVarAccessExprData(const IIRVarAccessExprData& other);

  bool operator==(const IIRVarAccessExprData&);
  bool operator!=(const IIRVarAccessExprData&);

  std::unique_ptr<ast::VarAccessExprData> clone() const override;

  boost::optional<int> AccessID;
};

struct IIRFieldAccessExprData : public ast::FieldAccessExprData {
  IIRFieldAccessExprData() = default;
  IIRFieldAccessExprData(const IIRFieldAccessExprData& other);

  bool operator==(const IIRFieldAccessExprData&);
  bool operator!=(const IIRFieldAccessExprData&);

  std::unique_ptr<ast::FieldAccessExprData> clone() const override;

  boost::optional<int> AccessID;
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
} // namespace iir
} // namespace dawn

#endif
