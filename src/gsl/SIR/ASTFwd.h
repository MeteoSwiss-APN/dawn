//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_SIR_ASTFWD_H
#define GSL_SIR_ASTFWD_H

#include "gsl/Support/Array.h"
#include "gsl/Support/SourceLocation.h"
#include "gsl/Support/Type.h"

namespace gsl {

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

} // namespace gsl

#endif
