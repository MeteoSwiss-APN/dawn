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

#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/Field.h"
#include "dawn/Support/Type.h"

#include <memory>

namespace dawn {
namespace astgen {

#define EXPR_CONSTRUCTOR_ALIAS(alias, expr)                                                        \
  template <typename... Args>                                                                      \
  decltype(auto) alias(Args&&... args) {                                                           \
    return std::make_shared<expr>(std::forward<Args>(args)...);                                    \
  }

#define STMT_CONSTRUCTOR_ALIAS(alias, maker)                                                       \
  template <typename... Args>                                                                      \
  decltype(auto) alias(Args&&... args) {                                                           \
    return maker(std::forward<Args>(args)...);                                                     \
  }

STMT_CONSTRUCTOR_ALIAS(expr, iir::makeExprStmt)
STMT_CONSTRUCTOR_ALIAS(ifstmt, iir::makeIfStmt)

EXPR_CONSTRUCTOR_ALIAS(assign, ast::AssignmentExpr)
EXPR_CONSTRUCTOR_ALIAS(binop, ast::BinaryOperator)
EXPR_CONSTRUCTOR_ALIAS(var, ast::VarAccessExpr)
EXPR_CONSTRUCTOR_ALIAS(field, ast::FieldAccessExpr)
EXPR_CONSTRUCTOR_ALIAS(lit, ast::LiteralAccessExpr)

template <typename... Stmts>
decltype(auto) block(Stmts&&... stmts) {
  return iir::makeBlockStmt(std::vector<std::shared_ptr<ast::Stmt>>{std::move(stmts)...});
}

inline decltype(auto) vardecl(const std::string& name, BuiltinTypeID type = BuiltinTypeID::Float) {
  return iir::makeVarDeclStmt(Type(type), name, 0, "=", ast::VarDeclStmt::InitList{});
}

template <typename T>
decltype(auto) lit(T&& value) {
  return std::make_shared<dawn::ast::LiteralAccessExpr>(
      std::to_string(std::forward<T>(value)),
      dawn::sir::Value::typeToBuiltinTypeID(
          dawn::sir::Value::TypeInfo<typename std::decay<T>::type>::Type));
}

template <typename... Args>
decltype(auto) global(Args&&... args) {
  auto expr = var(std::forward<Args>(args)...);
  expr->setIsExternal(true);
  return expr;
}

} // namespace astgen

} // namespace dawn
