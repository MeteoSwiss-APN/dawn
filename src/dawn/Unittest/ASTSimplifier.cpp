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

#include "dawn/Support/StringSwitch.h"
#include "dawn/Unittest/ASTSimplifier.h"

namespace dawn {

namespace astgen {

static Type stringToType(const std::string& typestring) {
  Type retval = StringSwitch<Type>(typestring)
                    .Case("int", BuiltinTypeID::Integer)
                    .Case("float", BuiltinTypeID::Float)
                    .Case("auto", BuiltinTypeID::Auto)
                    .Case("bool", BuiltinTypeID::Boolean)
                    .Default(BuiltinTypeID::Invalid);
  return retval;
}

std::shared_ptr<ExprStmt> expr(const std::shared_ptr<Expr>& expr) {
  return std::make_shared<ExprStmt>(expr);
}

std::shared_ptr<ReturnStmt> ret(const std::shared_ptr<Expr>& expr) {
  return std::make_shared<ReturnStmt>(expr);
}

std::shared_ptr<VarDeclStmt> vardecl(const std::string& type, const std::string& name,
                                     const std::shared_ptr<Expr>& init, const char* op) {

  return vecdecl(type, name, std::vector<std::shared_ptr<Expr>>({init}), 0, op);
}

std::shared_ptr<VarDeclStmt> vecdecl(const std::string& type, const std::string& name,
                                     std::vector<std::shared_ptr<Expr>> initList, int dimension,
                                     const char* op) {
  auto realtype = stringToType(type);
  return std::make_shared<VarDeclStmt>(realtype, name, dimension, op, initList);
}

std::shared_ptr<VerticalRegionDeclStmt>
verticalRegion(const std::shared_ptr<sir::VerticalRegion>& verticalRegion) {
  return std::make_shared<VerticalRegionDeclStmt>(verticalRegion);
}

std::shared_ptr<StencilCallDeclStmt> scdec(const std::shared_ptr<sir::StencilCall>& stencilCall) {
  return std::make_shared<StencilCallDeclStmt>(stencilCall);
}

std::shared_ptr<BoundaryConditionDeclStmt> boundaryCondition(const std::string& callee) {
  return std::make_shared<BoundaryConditionDeclStmt>(callee);
}

std::shared_ptr<IfStmt> ifstmt(const std::shared_ptr<Stmt>& condExpr,
                               const std::shared_ptr<Stmt>& thenStmt,
                               const std::shared_ptr<Stmt>& elseStmt) {
  return std::make_shared<IfStmt>(condExpr, thenStmt, elseStmt);
}

std::shared_ptr<UnaryOperator> unop(const std::shared_ptr<Expr>& operand, const char* op) {
  return std::make_shared<UnaryOperator>(operand, op);
}

std::shared_ptr<BinaryOperator> binop(const std::shared_ptr<Expr>& left, const char* op,
                                      const std::shared_ptr<Expr>& right) {
  return std::make_shared<BinaryOperator>(left, op, right);
}

std::shared_ptr<AssignmentExpr> assign(const std::shared_ptr<Expr>& left,
                                       const std::shared_ptr<Expr>& right, const char* op) {
  return std::make_shared<AssignmentExpr>(left, right, op);
}

std::shared_ptr<TernaryOperator> ternop(const std::shared_ptr<Expr>& cond,
                                        const std::shared_ptr<Expr>& left,
                                        const std::shared_ptr<Expr>& right) {
  return std::make_shared<TernaryOperator>(cond, left, right);
}

std::shared_ptr<FunCallExpr> fcall(const std::string& callee) {
  return std::make_shared<FunCallExpr>(callee);
}

std::shared_ptr<StencilFunCallExpr> sfcall(const std::string& callee) {
  return std::make_shared<StencilFunCallExpr>(callee);
}

std::shared_ptr<StencilFunArgExpr> arg(int direction, int offset, int argumentIndex) {
  return std::make_shared<StencilFunArgExpr>(direction, offset, argumentIndex);
}

std::shared_ptr<VarAccessExpr> var(const std::string& name, std::shared_ptr<Expr> index) {
  return std::make_shared<VarAccessExpr>(name, index);
}

std::shared_ptr<FieldAccessExpr> field(const std::string& name, Array3i offset, Array3i argumentMap,
                                       Array3i argumentOffset, bool negateOffset) {
  return std::make_shared<FieldAccessExpr>(name, offset, argumentMap, argumentOffset, negateOffset);
}

std::shared_ptr<LiteralAccessExpr> lit(const std::string& value, BuiltinTypeID builtinType) {
  return std::make_shared<LiteralAccessExpr>(value, builtinType);
}

} // namespace astgen

} // namespace dawn
