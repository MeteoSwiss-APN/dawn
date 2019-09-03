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

#ifndef DAWN_AST_ASTFWD_H
#define DAWN_AST_ASTFWD_H

namespace dawn {
namespace ast {
template <typename DataTraits>
class AST;

template <typename DataTraits>
class Stmt;
template <typename DataTraits>
class BlockStmt;
template <typename DataTraits>
class ExprStmt;
template <typename DataTraits>
class ReturnStmt;
template <typename DataTraits>
class VarDeclStmt;
template <typename DataTraits>
class StencilCallDeclStmt;
template <typename DataTraits>
class BoundaryConditionDeclStmt;
template <typename DataTraits>
class IfStmt;

template <typename DataTraits>
class Expr;
template <typename DataTraits>
class NOPExpr;
template <typename DataTraits>
class UnaryOperator;
template <typename DataTraits>
class BinaryOperator;
template <typename DataTraits>
class AssignmentExpr;
template <typename DataTraits>
class TernaryOperator;
template <typename DataTraits>
class FunCallExpr;
template <typename DataTraits>
class StencilFunCallExpr;
template <typename DataTraits>
class StencilFunArgExpr;
template <typename DataTraits>
class VarAccessExpr;
template <typename DataTraits>
class FieldAccessExpr;
template <typename DataTraits>
class LiteralAccessExpr;

class ASTHelper;
template <typename DataTraits>
class ASTVisitor;
} // namespace ast
} // namespace dawn

#endif
