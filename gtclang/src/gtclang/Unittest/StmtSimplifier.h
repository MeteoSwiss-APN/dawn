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

#include "dawn/SIR/ASTExpr.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/SIR.h"
#include <memory>

namespace gtclang {

namespace sirgen {

/// @brief Helper Class to write a concatination of statements into a block statement
class BlockWriter {
public:
  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<dawn::ast::Stmt>& statement, Args&&... args) {
    storage_.push_back(statement);
    recursiveBlock(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<dawn::ast::Expr>& expression, Args&&... args) {
    recursiveBlock(dawn::sir::makeExprStmt(expression), std::forward<Args>(args)...);
  }

  void recursiveBlock() {}

  const std::vector<std::shared_ptr<dawn::ast::Stmt>>& createVec() { return storage_; }

  template <typename... Args>
  const std::vector<std::shared_ptr<dawn::ast::Stmt>>&
  createVec(const std::shared_ptr<dawn::ast::Stmt>& statement, Args&&... args) {
    recursiveBlock(statement, std::forward<Args>(args)...);
    return storage_;
  }
  template <typename... Args>
  const std::vector<std::shared_ptr<dawn::ast::Stmt>>&
  createVec(const std::shared_ptr<dawn::ast::Expr>& expr, Args&&... args) {
    recursiveBlock(expr, std::forward<Args>(args)...);
    return storage_;
  }

private:
  std::vector<std::shared_ptr<dawn::ast::Stmt>> storage_;
};

/// @brief simplification for generating SIR in memory
/// This group of statements allows for a simplyfied notation to generate in-memory SIRs for testing
/// puropses. It can be used to describe simple operations or blocks of operations in a human
/// readable way like
/// @code{.cpp}
/// assign(var("a"), binop(var("b"),"+",var("c")))
/// @endcode
/// @ingroup unittest
/// @{
template <typename... Args>
std::shared_ptr<dawn::ast::BlockStmt> block(Args&&... args) {
  BlockWriter bw;
  auto vec = bw.createVec(std::forward<Args>(args)...);
  return dawn::sir::makeBlockStmt(vec);
}

std::shared_ptr<dawn::ast::ExprStmt> expr(const std::shared_ptr<dawn::ast::Expr>& expr);
std::shared_ptr<dawn::ast::ReturnStmt> ret(const std::shared_ptr<dawn::ast::Expr>& expr);
std::shared_ptr<dawn::ast::VarDeclStmt> vardecl(const std::string& type, const std::string& name,
                                                const std::shared_ptr<dawn::ast::Expr>& init,
                                                const std::string op = "=");
std::shared_ptr<dawn::ast::VarDeclStmt>
vecdecl(const std::string& type, const std::string& name,
        std::vector<std::shared_ptr<dawn::ast::Expr>> initList, int dimension = 0,
        const std::string op = "=");
std::shared_ptr<dawn::ast::VerticalRegionDeclStmt>
verticalRegion(const std::shared_ptr<dawn::sir::VerticalRegion>& verticalRegion);
std::shared_ptr<dawn::ast::StencilCallDeclStmt>
scdec(const std::shared_ptr<dawn::ast::StencilCall>& stencilCall);
std::shared_ptr<dawn::ast::BoundaryConditionDeclStmt> boundaryCondition(const std::string& callee);
std::shared_ptr<dawn::ast::IfStmt>
ifstmt(const std::shared_ptr<dawn::ast::Stmt>& condExpr,
       const std::shared_ptr<dawn::ast::Stmt>& thenStmt,
       const std::shared_ptr<dawn::ast::Stmt>& elseStmt = nullptr);
std::shared_ptr<dawn::ast::UnaryOperator> unop(const std::shared_ptr<dawn::ast::Expr>& operand,
                                               const std::string op);
std::shared_ptr<dawn::ast::BinaryOperator> binop(const std::shared_ptr<dawn::ast::Expr>& left,
                                                 const std::string op,
                                                 const std::shared_ptr<dawn::ast::Expr>& right);
std::shared_ptr<dawn::ast::AssignmentExpr> assign(const std::shared_ptr<dawn::ast::Expr>& left,
                                                  const std::shared_ptr<dawn::ast::Expr>& right,
                                                  const std::string op = "=");
std::shared_ptr<dawn::ast::TernaryOperator> ternop(const std::shared_ptr<dawn::ast::Expr>& cond,
                                                   const std::shared_ptr<dawn::ast::Expr>& left,
                                                   const std::shared_ptr<dawn::ast::Expr>& right);
std::shared_ptr<dawn::ast::FunCallExpr> fcall(const std::string& callee);
std::shared_ptr<dawn::ast::StencilFunCallExpr> sfcall(const std::string& calee);
std::shared_ptr<dawn::ast::StencilFunArgExpr> arg(int direction, int offset, int argumentIndex);
std::shared_ptr<dawn::ast::VarAccessExpr> var(const std::string& name,
                                              std::shared_ptr<dawn::ast::Expr> index = nullptr);
std::shared_ptr<dawn::ast::FieldAccessExpr> field(const std::string& name);
std::shared_ptr<dawn::ast::FieldAccessExpr>
field(const std::string& name, dawn::Array3i offset,
      dawn::Array3i argumentMap = dawn::Array3i{{-1, -1, -1}},
      dawn::Array3i argumentOffset = dawn::Array3i{{0, 0, 0}}, bool negateOffset = false);
std::shared_ptr<dawn::ast::LiteralAccessExpr>
lit(const std::string& value, dawn::BuiltinTypeID builtinType = dawn::BuiltinTypeID::Integer);
std::shared_ptr<dawn::ast::LiteralAccessExpr>
lit(const char* value, dawn::BuiltinTypeID builtinType = dawn::BuiltinTypeID::Integer);

/// @}

} // namespace sirgen

} // namespace gtclang