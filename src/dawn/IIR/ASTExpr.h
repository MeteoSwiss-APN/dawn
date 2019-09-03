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
#include "dawn/IIR/ASTData.h"

namespace dawn {
namespace iir {
using Expr = ast::Expr<IIRASTData>;
using NOPExpr = ast::NOPExpr<IIRASTData>;
using UnaryOperator = ast::UnaryOperator<IIRASTData>;
using BinaryOperator = ast::BinaryOperator<IIRASTData>;
using AssignmentExpr = ast::AssignmentExpr<IIRASTData>;
using TernaryOperator = ast::TernaryOperator<IIRASTData>;
using FunCallExpr = ast::FunCallExpr<IIRASTData>;
using StencilFunCallExpr = ast::StencilFunCallExpr<IIRASTData>;
using StencilFunArgExpr = ast::StencilFunArgExpr<IIRASTData>;
using VarAccessExpr = ast::VarAccessExpr<IIRASTData>;
using FieldAccessExpr = ast::FieldAccessExpr<IIRASTData>;
using LiteralAccessExpr = ast::LiteralAccessExpr<IIRASTData>;
} // namespace iir
} // namespace dawn

#endif
