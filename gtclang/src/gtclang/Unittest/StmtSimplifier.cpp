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

#include "StmtSimplifier.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/Unreachable.h"

namespace gtclang {
namespace sirgen {

static dawn::Type stringToType(const std::string& typestring) {
  dawn::Type retval = dawn::StringSwitch<dawn::Type>(typestring)
                          .Case("int", dawn::BuiltinTypeID::Integer)
                          .Case("float", dawn::BuiltinTypeID::Float)
                          .Case("auto", dawn::BuiltinTypeID::Auto)
                          .Case("bool", dawn::BuiltinTypeID::Boolean)
                          .Default(dawn::BuiltinTypeID::Invalid);
  if(!retval.isBuiltinType())
    dawn_unreachable("wrong type");
  return retval;
}

std::shared_ptr<dawn::ast::ExprStmt> expr(const std::shared_ptr<dawn::ast::Expr>& expr) {
  return dawn::sir::makeExprStmt(expr);
}

std::shared_ptr<dawn::ast::ReturnStmt> ret(const std::shared_ptr<dawn::ast::Expr>& expr) {
  return dawn::sir::makeReturnStmt(expr);
}

std::shared_ptr<dawn::ast::VarDeclStmt> vardecl(const std::string& type, const std::string& name,
                                                const std::shared_ptr<dawn::ast::Expr>& init,
                                                const std::string op) {

  return vecdecl(type, name, std::vector<std::shared_ptr<dawn::ast::Expr>>({init}), 0, op);
}

std::shared_ptr<dawn::ast::VarDeclStmt>
vecdecl(const std::string& type, const std::string& name,
        std::vector<std::shared_ptr<dawn::ast::Expr>> initList, int dimension,
        const std::string op) {
  auto realtype = stringToType(type);
  return dawn::sir::makeVarDeclStmt(realtype, name, dimension, op, initList);
}

std::shared_ptr<dawn::ast::VerticalRegionDeclStmt>
verticalRegion(const std::shared_ptr<dawn::sir::VerticalRegion>& verticalRegion) {
  return dawn::sir::makeVerticalRegionDeclStmt(verticalRegion);
}

std::shared_ptr<dawn::ast::StencilCallDeclStmt>
scdec(const std::shared_ptr<dawn::ast::StencilCall>& stencilCall) {
  return dawn::sir::makeStencilCallDeclStmt(stencilCall);
}

std::shared_ptr<dawn::ast::BoundaryConditionDeclStmt> boundaryCondition(const std::string& callee) {
  return dawn::sir::makeBoundaryConditionDeclStmt(callee);
}

std::shared_ptr<dawn::ast::IfStmt> ifstmt(const std::shared_ptr<dawn::ast::Stmt>& condExpr,
                                          const std::shared_ptr<dawn::ast::Stmt>& thenStmt,
                                          const std::shared_ptr<dawn::ast::Stmt>& elseStmt) {
  return dawn::sir::makeIfStmt(condExpr, thenStmt, elseStmt);
}

std::shared_ptr<dawn::ast::UnaryOperator> unop(const std::shared_ptr<dawn::ast::Expr>& operand,
                                               const std::string op) {
  return std::make_shared<dawn::ast::UnaryOperator>(operand, op);
}

std::shared_ptr<dawn::ast::BinaryOperator> binop(const std::shared_ptr<dawn::ast::Expr>& left,
                                                 const std::string op,
                                                 const std::shared_ptr<dawn::ast::Expr>& right) {
  return std::make_shared<dawn::ast::BinaryOperator>(left, op, right);
}

std::shared_ptr<dawn::ast::AssignmentExpr> assign(const std::shared_ptr<dawn::ast::Expr>& left,
                                                  const std::shared_ptr<dawn::ast::Expr>& right,
                                                  const std::string op) {
  return std::make_shared<dawn::ast::AssignmentExpr>(left, right, op);
}

std::shared_ptr<dawn::ast::TernaryOperator> ternop(const std::shared_ptr<dawn::ast::Expr>& cond,
                                                   const std::shared_ptr<dawn::ast::Expr>& left,
                                                   const std::shared_ptr<dawn::ast::Expr>& right) {
  return std::make_shared<dawn::ast::TernaryOperator>(cond, left, right);
}

std::shared_ptr<dawn::ast::FunCallExpr> fcall(const std::string& callee) {
  return std::make_shared<dawn::ast::FunCallExpr>(callee);
}

std::shared_ptr<dawn::ast::StencilFunCallExpr> sfcall(const std::string& calee) {
  return std::make_shared<dawn::ast::StencilFunCallExpr>(calee);
}

std::shared_ptr<dawn::ast::StencilFunArgExpr> arg(int direction, int offset, int argumentIndex) {
  return std::make_shared<dawn::ast::StencilFunArgExpr>(direction, offset, argumentIndex);
}

std::shared_ptr<dawn::ast::VarAccessExpr> var(const std::string& name,
                                              std::shared_ptr<dawn::ast::Expr> index) {
  return std::make_shared<dawn::ast::VarAccessExpr>(name, index);
}

std::shared_ptr<dawn::ast::FieldAccessExpr> field(const std::string& name) {
  return std::make_shared<dawn::ast::FieldAccessExpr>(name, dawn::ast::Offsets{});
}

std::shared_ptr<dawn::ast::FieldAccessExpr> field(const std::string& name, dawn::Array3i offset,
                                                  dawn::Array3i argumentMap,
                                                  dawn::Array3i argumentOffset, bool negateOffset) {
  return std::make_shared<dawn::ast::FieldAccessExpr>(
      name, dawn::ast::Offsets{dawn::ast::cartesian, offset}, argumentMap, argumentOffset,
      negateOffset);
}
std::shared_ptr<dawn::ast::LiteralAccessExpr> lit(const std::string& value,
                                                  dawn::BuiltinTypeID builtinType) {
  return std::make_shared<dawn::ast::LiteralAccessExpr>(value, builtinType);
}
std::shared_ptr<dawn::ast::LiteralAccessExpr> lit(const char* value,
                                                  dawn::BuiltinTypeID builtinType) {
  return lit(std::string(value), builtinType);
}

} // namespace sirgen

} // namespace gtclang
