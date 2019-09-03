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

#include "dawn/Unittest/ASTSimplifier.h"
#include "dawn/Support/StringSwitch.h"

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

std::shared_ptr<sir::ExprStmt> expr(const std::shared_ptr<sir::Expr>& expr) {
  return std::make_shared<sir::ExprStmt>(expr);
}

std::shared_ptr<sir::ReturnStmt> ret(const std::shared_ptr<sir::Expr>& expr) {
  return std::make_shared<sir::ReturnStmt>(expr);
}

std::shared_ptr<sir::VarDeclStmt> vardecl(const std::string& type, const std::string& name,
                                          const std::shared_ptr<sir::Expr>& init, const char* op) {

  return vecdecl(type, name, std::vector<std::shared_ptr<sir::Expr>>({init}), 0, op);
}

std::shared_ptr<sir::VarDeclStmt> vecdecl(const std::string& type, const std::string& name,
                                          std::vector<std::shared_ptr<sir::Expr>> initList,
                                          int dimension, const char* op) {
  auto realtype = stringToType(type);
  return std::make_shared<sir::VarDeclStmt>(realtype, name, dimension, op, initList);
}

std::shared_ptr<sir::VerticalRegionDeclStmt>
verticalRegion(const std::shared_ptr<sir::AST>& ast,
               const std::shared_ptr<sir::VerticalRegion>& verticalRegion) {
  return std::make_shared<sir::VerticalRegionDeclStmt>(ast, verticalRegion);
}

std::shared_ptr<sir::StencilCallDeclStmt>
scdec(const std::shared_ptr<ast::StencilCall>& stencilCall) {
  return std::make_shared<sir::StencilCallDeclStmt>(stencilCall);
}

std::shared_ptr<sir::BoundaryConditionDeclStmt> boundaryCondition(const std::string& callee) {
  return std::make_shared<sir::BoundaryConditionDeclStmt>(callee);
}

std::shared_ptr<sir::IfStmt> ifstmt(const std::shared_ptr<sir::Stmt>& condExpr,
                                    const std::shared_ptr<sir::Stmt>& thenStmt,
                                    const std::shared_ptr<sir::Stmt>& elseStmt) {
  return std::make_shared<sir::IfStmt>(condExpr, thenStmt, elseStmt);
}

std::shared_ptr<sir::UnaryOperator> unop(const std::shared_ptr<sir::Expr>& operand,
                                         const char* op) {
  return std::make_shared<sir::UnaryOperator>(operand, op);
}

std::shared_ptr<sir::BinaryOperator> binop(const std::shared_ptr<sir::Expr>& left, const char* op,
                                           const std::shared_ptr<sir::Expr>& right) {
  return std::make_shared<sir::BinaryOperator>(left, op, right);
}

std::shared_ptr<sir::AssignmentExpr> assign(const std::shared_ptr<sir::Expr>& left,
                                            const std::shared_ptr<sir::Expr>& right,
                                            const char* op) {
  return std::make_shared<sir::AssignmentExpr>(left, right, op);
}

std::shared_ptr<sir::TernaryOperator> ternop(const std::shared_ptr<sir::Expr>& cond,
                                             const std::shared_ptr<sir::Expr>& left,
                                             const std::shared_ptr<sir::Expr>& right) {
  return std::make_shared<sir::TernaryOperator>(cond, left, right);
}

std::shared_ptr<sir::FunCallExpr> fcall(const std::string& callee) {
  return std::make_shared<sir::FunCallExpr>(callee);
}

std::shared_ptr<sir::StencilFunCallExpr> sfcall(const std::string& callee) {
  return std::make_shared<sir::StencilFunCallExpr>(callee);
}

std::shared_ptr<sir::StencilFunArgExpr> arg(int direction, int offset, int argumentIndex) {
  return std::make_shared<sir::StencilFunArgExpr>(direction, offset, argumentIndex);
}

std::shared_ptr<sir::VarAccessExpr> var(const std::string& name, std::shared_ptr<sir::Expr> index) {
  return std::make_shared<sir::VarAccessExpr>(name, index);
}

std::shared_ptr<sir::FieldAccessExpr> field(const std::string& name, Array3i offset,
                                            Array3i argumentMap, Array3i argumentOffset,
                                            bool negateOffset) {
  return std::make_shared<sir::FieldAccessExpr>(name, offset, argumentMap, argumentOffset,
                                                negateOffset);
}

std::shared_ptr<sir::LiteralAccessExpr> lit(const std::string& value, BuiltinTypeID builtinType) {
  return std::make_shared<sir::LiteralAccessExpr>(value, builtinType);
}

} // namespace astgen

} // namespace dawn
