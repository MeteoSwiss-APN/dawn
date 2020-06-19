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

namespace dawn {
namespace ast {
class AST;

struct StmtData;

class Stmt;
class BlockStmt;
class ExprStmt;
class ReturnStmt;
class VarDeclStmt;
class VerticalRegionDeclStmt;
class StencilCallDeclStmt;
class BoundaryConditionDeclStmt;
class IfStmt;
class LoopStmt;

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
class ReductionOverNeighborExpr;

class ASTHelper;
class ASTVisitor; //   Compiler complains if declared as class
} // namespace ast
} // namespace dawn
