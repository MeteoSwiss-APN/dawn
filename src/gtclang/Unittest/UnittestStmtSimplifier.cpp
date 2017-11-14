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

std::shared_ptr<dawn::ExprStmt> expr(const std::shared_ptr<dawn::Expr>& expr) {
  return std::make_shared<dawn::ExprStmt>(expr);
}

std::shared_ptr<dawn::ReturnStmt> ret(const std::shared_ptr<dawn::Expr>& expr) {
  return std::make_shared<dawn::ReturnStmt>(expr);
}

std::shared_ptr<dawn::VarDeclStmt> vardecl(const std::string& type, const std::string& name,
                                           const std::shared_ptr<dawn::Expr>& init,
                                           const char* op) {

  return vecdecl(type, name, std::vector<std::shared_ptr<dawn::Expr>>({init}), 0, op);
}

std::shared_ptr<dawn::VarDeclStmt> vecdecl(const std::string& type, const std::string& name,
                                           std::vector<std::shared_ptr<dawn::Expr>> initList,
                                           int dimension, const char* op) {
  auto realtype = stringToType(type);
  return std::make_shared<dawn::VarDeclStmt>(realtype, name, dimension, op, initList);
}

std::shared_ptr<dawn::VerticalRegionDeclStmt>
verticalRegion(const std::shared_ptr<dawn::sir::VerticalRegion>& verticalRegion) {
  return std::make_shared<dawn::VerticalRegionDeclStmt>(verticalRegion);
}

std::shared_ptr<dawn::StencilCallDeclStmt>
scdec(const std::shared_ptr<dawn::sir::StencilCall>& stencilCall) {
  return std::make_shared<dawn::StencilCallDeclStmt>(stencilCall);
}

std::shared_ptr<dawn::BoundaryConditionDeclStmt> boundaryCondition(const std::string& callee) {
  return std::make_shared<dawn::BoundaryConditionDeclStmt>(callee);
}

std::shared_ptr<dawn::IfStmt> ifstmt(const std::shared_ptr<dawn::Stmt>& condExpr,
                                     const std::shared_ptr<dawn::Stmt>& thenStmt,
                                     const std::shared_ptr<dawn::Stmt>& elseStmt) {
  return std::make_shared<dawn::IfStmt>(condExpr, thenStmt, elseStmt);
}

std::shared_ptr<dawn::UnaryOperator> unop(const std::shared_ptr<dawn::Expr>& operand,
                                          const char* op) {
  return std::make_shared<dawn::UnaryOperator>(operand, op);
}

std::shared_ptr<dawn::BinaryOperator> binop(const std::shared_ptr<dawn::Expr>& left, const char* op,
                                            const std::shared_ptr<dawn::Expr>& right) {
  return std::make_shared<dawn::BinaryOperator>(left, op, right);
}

std::shared_ptr<dawn::AssignmentExpr> assign(const std::shared_ptr<dawn::Expr>& left,
                                             const std::shared_ptr<dawn::Expr>& right,
                                             const char* op) {
  return std::make_shared<dawn::AssignmentExpr>(left, right, op);
}

std::shared_ptr<dawn::TernaryOperator> ternop(const std::shared_ptr<dawn::Expr>& cond,
                                              const std::shared_ptr<dawn::Expr>& left,
                                              const std::shared_ptr<dawn::Expr>& right) {
  return std::make_shared<dawn::TernaryOperator>(cond, left, right);
}

std::shared_ptr<dawn::FunCallExpr> fcall(const std::string& callee) {
  return std::make_shared<dawn::FunCallExpr>(callee);
}

std::shared_ptr<dawn::StencilFunCallExpr> sfcall(const std::string& calee) {
  return std::make_shared<dawn::StencilFunCallExpr>(calee);
}

std::shared_ptr<dawn::StencilFunArgExpr> arg(int direction, int offset, int argumentIndex) {
  return std::make_shared<dawn::StencilFunArgExpr>(direction, offset, argumentIndex);
}

std::shared_ptr<dawn::VarAccessExpr> var(const std::string& name,
                                         std::shared_ptr<dawn::Expr> index) {
  return std::make_shared<dawn::VarAccessExpr>(name, index);
}

std::shared_ptr<dawn::FieldAccessExpr> field(const std::string& name, dawn::Array3i offset,
                                             dawn::Array3i argumentMap,
                                             dawn::Array3i argumentOffset, bool negateOffset) {
  return std::make_shared<dawn::FieldAccessExpr>(name, offset, argumentMap, argumentOffset,
                                                 negateOffset);
}

std::shared_ptr<dawn::LiteralAccessExpr> lit(const std::string& value,
                                             dawn::BuiltinTypeID builtinType) {
  return std::make_shared<dawn::LiteralAccessExpr>(value, builtinType);
}

} // namespace sirgen

} // namespace gtclang
