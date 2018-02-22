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

#include "dawn/Support/Array.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Support/Type.h"

namespace dawn {

class AST;

class Stmt;
class BlockStmt;
class ExprStmt;
class ReturnStmt;
class VarDeclStmt;
class VerticalRegionDeclStmt;
class StencilCallDeclStmt;
class BoundaryConditionDeclStmt;
class IfStmt;

class Expr;
class NOPExpr;
class UnaryOperator;
class BinaryOperator;
class AssignmentExpr;
class TernaryOperator;
class FunCallExpr;
class StencilFunCallExpr;
class StencilFunArgExpr;
class VarAccessExpr;
class FieldAccessExpr;
class LiteralAccessExpr;

class ASTVisitor;

} // namespace dawn

#endif
