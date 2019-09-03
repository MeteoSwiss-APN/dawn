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

#ifndef DAWN_IIR_ASTFWD_H
#define DAWN_IIR_ASTFWD_H

#include "dawn/AST/ASTFwd.h"
#include "dawn/IIR/ASTData.h"

namespace dawn {
namespace iir {
using AST = ast::AST<IIRASTData>;

using Stmt = ast::Stmt<IIRASTData>;
using BlockStmt = ast::BlockStmt<IIRASTData>;
using ExprStmt = ast::ExprStmt<IIRASTData>;
using ReturnStmt = ast::ReturnStmt<IIRASTData>;
using VarDeclStmt = ast::VarDeclStmt<IIRASTData>;
using StencilCallDeclStmt = ast::StencilCallDeclStmt<IIRASTData>;
using BoundaryConditionDeclStmt = ast::BoundaryConditionDeclStmt<IIRASTData>;
using IfStmt = ast::IfStmt<IIRASTData>;

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

using ASTHelper = ast::ASTHelper;
using ASTVisitor = ast::ASTVisitor<IIRASTData>;
} // namespace iir
} // namespace dawn

#endif
