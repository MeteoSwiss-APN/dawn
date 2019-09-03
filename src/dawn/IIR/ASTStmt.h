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
#include "dawn/IIR/ASTData.h"

namespace dawn {
namespace iir {
using Stmt = ast::Stmt<IIRASTData>;
using BlockStmt = ast::BlockStmt<IIRASTData>;
using ExprStmt = ast::ExprStmt<IIRASTData>;
using ReturnStmt = ast::ReturnStmt<IIRASTData>;
using VarDeclStmt = ast::VarDeclStmt<IIRASTData>;
using StencilCallDeclStmt = ast::StencilCallDeclStmt<IIRASTData>;
using BoundaryConditionDeclStmt = ast::BoundaryConditionDeclStmt<IIRASTData>;
using IfStmt = ast::IfStmt<IIRASTData>;

} // namespace iir
} // namespace dawn

#endif
