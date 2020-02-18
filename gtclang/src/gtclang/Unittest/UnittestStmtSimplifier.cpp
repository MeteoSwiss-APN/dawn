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

#include "gtclang/Unittest/UnittestStmtSimplifier.h"
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

std::shared_ptr<dawn::sir::ExprStmt> expr(const std::shared_ptr<dawn::sir::Expr>& expr) {
  return dawn::sir::makeExprStmt(expr);
}

std::shared_ptr<dawn::sir::ReturnStmt> ret(const std::shared_ptr<dawn::sir::Expr>& expr) {
  return dawn::sir::makeReturnStmt(expr);
}

std::shared_ptr<dawn::sir::VarDeclStmt> vardecl(const std::string& type, const std::string& name,
                                                const std::shared_ptr<dawn::sir::Expr>& init,
                                                const std::string op) {

  return vecdecl(type, name, std::vector<std::shared_ptr<dawn::sir::Expr>>({init}), 0, op);
}

std::shared_ptr<dawn::sir::VarDeclStmt>
vecdecl(const std::string& type, const std::string& name,
        std::vector<std::shared_ptr<dawn::sir::Expr>> initList, int dimension,
        const std::string op) {
  auto realtype = stringToType(type);
  return dawn::sir::makeVarDeclStmt(realtype, name, dimension, op, initList);
}

std::shared_ptr<dawn::sir::VerticalRegionDeclStmt>
verticalRegion(const std::shared_ptr<dawn::sir::VerticalRegion>& verticalRegion) {
  return dawn::sir::makeVerticalRegionDeclStmt(verticalRegion);
}

std::shared_ptr<dawn::sir::StencilCallDeclStmt>
scdec(const std::shared_ptr<dawn::ast::StencilCall>& stencilCall) {
  return dawn::sir::makeStencilCallDeclStmt(stencilCall);
}

std::shared_ptr<dawn::sir::BoundaryConditionDeclStmt> boundaryCondition(const std::string& callee) {
  return dawn::sir::makeBoundaryConditionDeclStmt(callee);
}

std::shared_ptr<dawn::sir::IfStmt> ifstmt(const std::shared_ptr<dawn::sir::Stmt>& condExpr,
                                          const std::shared_ptr<dawn::sir::Stmt>& thenStmt,
                                          const std::shared_ptr<dawn::sir::Stmt>& elseStmt) {
  return dawn::sir::makeIfStmt(condExpr, thenStmt, elseStmt);
}

std::shared_ptr<dawn::sir::UnaryOperator> unop(const std::shared_ptr<dawn::sir::Expr>& operand,
                                               const std::string op) {
  return std::make_shared<dawn::sir::UnaryOperator>(operand, op);
}

std::shared_ptr<dawn::sir::BinaryOperator> binop(const std::shared_ptr<dawn::sir::Expr>& left,
                                                 const std::string op,
                                                 const std::shared_ptr<dawn::sir::Expr>& right) {
  return std::make_shared<dawn::sir::BinaryOperator>(left, op, right);
}

std::shared_ptr<dawn::sir::AssignmentExpr> assign(const std::shared_ptr<dawn::sir::Expr>& left,
                                                  const std::shared_ptr<dawn::sir::Expr>& right,
                                                  const std::string op) {
  return std::make_shared<dawn::sir::AssignmentExpr>(left, right, op);
}

std::shared_ptr<dawn::sir::TernaryOperator> ternop(const std::shared_ptr<dawn::sir::Expr>& cond,
                                                   const std::shared_ptr<dawn::sir::Expr>& left,
                                                   const std::shared_ptr<dawn::sir::Expr>& right) {
  return std::make_shared<dawn::sir::TernaryOperator>(cond, left, right);
}

std::shared_ptr<dawn::sir::FunCallExpr> fcall(const std::string& callee) {
  return std::make_shared<dawn::sir::FunCallExpr>(callee);
}

std::shared_ptr<dawn::sir::StencilFunCallExpr> sfcall(const std::string& calee) {
  return std::make_shared<dawn::sir::StencilFunCallExpr>(calee);
}

std::shared_ptr<dawn::sir::StencilFunArgExpr> arg(int direction, int offset, int argumentIndex) {
  return std::make_shared<dawn::sir::StencilFunArgExpr>(direction, offset, argumentIndex);
}

std::shared_ptr<dawn::sir::VarAccessExpr> var(const std::string& name,
                                              std::shared_ptr<dawn::sir::Expr> index) {
  return std::make_shared<dawn::sir::VarAccessExpr>(name, index);
}

std::shared_ptr<dawn::sir::FieldAccessExpr> field(const std::string& name, dawn::Array3i offset,
                                                  dawn::Array3i argumentMap,
                                                  dawn::Array3i argumentOffset, bool negateOffset) {
  return std::make_shared<dawn::sir::FieldAccessExpr>(
      name, dawn::ast::Offsets{dawn::ast::cartesian, offset}, argumentMap, argumentOffset,
      negateOffset);
}

std::shared_ptr<dawn::sir::LiteralAccessExpr> lit(const std::string& value,
                                                  dawn::BuiltinTypeID builtinType) {
  return std::make_shared<dawn::sir::LiteralAccessExpr>(value, builtinType);
}

} // namespace sirgen

} // namespace gtclang
