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

#ifndef DAWN_SIR_ASTFWD_H
#define DAWN_SIR_ASTFWD_H

#include "dawn/AST/ASTFwd.h"
#include "dawn/SIR/ASTData.h"

namespace dawn {
namespace sir {
using AST = ast::AST<SIRASTData>;

using Stmt = ast::Stmt<SIRASTData>;
using BlockStmt = ast::BlockStmt<SIRASTData>;
using ExprStmt = ast::ExprStmt<SIRASTData>;
using ReturnStmt = ast::ReturnStmt<SIRASTData>;
using VarDeclStmt = ast::VarDeclStmt<SIRASTData>;
using StencilCallDeclStmt = ast::StencilCallDeclStmt<SIRASTData>;
using BoundaryConditionDeclStmt = ast::BoundaryConditionDeclStmt<SIRASTData>;
using IfStmt = ast::IfStmt<SIRASTData>;

class VerticalRegionDeclStmt;

using Expr = ast::Expr<SIRASTData>;
using NOPExpr = ast::NOPExpr<SIRASTData>;
using UnaryOperator = ast::UnaryOperator<SIRASTData>;
using BinaryOperator = ast::BinaryOperator<SIRASTData>;
using AssignmentExpr = ast::AssignmentExpr<SIRASTData>;
using TernaryOperator = ast::TernaryOperator<SIRASTData>;
using FunCallExpr = ast::FunCallExpr<SIRASTData>;
using StencilFunCallExpr = ast::StencilFunCallExpr<SIRASTData>;
using StencilFunArgExpr = ast::StencilFunArgExpr<SIRASTData>;
using VarAccessExpr = ast::VarAccessExpr<SIRASTData>;
using FieldAccessExpr = ast::FieldAccessExpr<SIRASTData>;
using LiteralAccessExpr = ast::LiteralAccessExpr<SIRASTData>;

using ASTHelper = ast::ASTHelper;
class ASTVisitor;
} // namespace sir
} // namespace dawn

#endif
