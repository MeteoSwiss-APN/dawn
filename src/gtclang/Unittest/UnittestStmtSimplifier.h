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

#ifndef DAWN_UNITTEST_UNITTESTSTMTSIMPLYFIER_H
#define DAWN_UNITTEST_UNITTESTSTMTSIMPLYFIER_H

#include "dawn/SIR/ASTExpr.h"
#include "dawn/SIR/ASTStmt.h"
#include <memory>

namespace gtclang {

namespace sirgen {

/// @brief Helper Class to write a concatination of statements into a block statement
class BlockWriter {
public:
  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<dawn::sir::Stmt>& statement, Args&&... args) {
    storage_.push_back(statement);
    recursiveBlock(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<dawn::sir::Expr>& expression, Args&&... args) {
    recursiveBlock(std::make_shared<dawn::sir::ExprStmt>(expression), std::forward<Args>(args)...);
  }

  void recursiveBlock() {}

  template <typename... Args>
  const std::vector<std::shared_ptr<dawn::sir::Stmt>>&
  createVec(const std::shared_ptr<dawn::sir::Stmt>& statement, Args&&... args) {
    recursiveBlock(statement, std::forward<Args>(args)...);
    return storage_;
  }
  template <typename... Args>
  const std::vector<std::shared_ptr<dawn::sir::Stmt>>& createVec(const std::shared_ptr<dawn::sir::Expr>& expr,
                                                            Args&&... args) {
    recursiveBlock(expr, std::forward<Args>(args)...);
    return storage_;
  }

private:
  std::vector<std::shared_ptr<dawn::sir::Stmt>> storage_;
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
std::shared_ptr<dawn::sir::BlockStmt> block(Args&&... args) {
  BlockWriter bw;
  auto vec = bw.createVec(std::forward<Args>(args)...);
  return std::make_shared<dawn::sir::BlockStmt>(vec);
}

std::shared_ptr<dawn::sir::ExprStmt> expr(const std::shared_ptr<dawn::sir::Expr>& expr);
std::shared_ptr<dawn::sir::ReturnStmt> ret(const std::shared_ptr<dawn::sir::Expr>& expr);
std::shared_ptr<dawn::sir::VarDeclStmt> vardecl(const std::string& type, const std::string& name,
                                           const std::shared_ptr<dawn::sir::Expr>& init,
                                           const char* op = "=");
std::shared_ptr<dawn::sir::VarDeclStmt> vecdecl(const std::string& type, const std::string& name,
                                           std::vector<std::shared_ptr<dawn::sir::Expr>> initList,
                                           int dimension = 0, const char* op = "=");
std::shared_ptr<dawn::sir::VerticalRegionDeclStmt>
verticalRegion(const std::shared_ptr<dawn::sir::VerticalRegion>& verticalRegion);
std::shared_ptr<dawn::sir::StencilCallDeclStmt>
scdec(const std::shared_ptr<dawn::ast::StencilCall>& stencilCall);
std::shared_ptr<dawn::sir::BoundaryConditionDeclStmt> boundaryCondition(const std::string& callee);
std::shared_ptr<dawn::sir::IfStmt> ifstmt(const std::shared_ptr<dawn::sir::Stmt>& condExpr,
                                     const std::shared_ptr<dawn::sir::Stmt>& thenStmt,
                                     const std::shared_ptr<dawn::sir::Stmt>& elseStmt = nullptr);
std::shared_ptr<dawn::sir::UnaryOperator> unop(const std::shared_ptr<dawn::sir::Expr>& operand,
                                          const char* op);
std::shared_ptr<dawn::sir::BinaryOperator> binop(const std::shared_ptr<dawn::sir::Expr>& left, const char* op,
                                            const std::shared_ptr<dawn::sir::Expr>& right);
std::shared_ptr<dawn::sir::AssignmentExpr> assign(const std::shared_ptr<dawn::sir::Expr>& left,
                                             const std::shared_ptr<dawn::sir::Expr>& right,
                                             const char* op = "=");
std::shared_ptr<dawn::sir::TernaryOperator> ternop(const std::shared_ptr<dawn::sir::Expr>& cond,
                                              const std::shared_ptr<dawn::sir::Expr>& left,
                                              const std::shared_ptr<dawn::sir::Expr>& right);
std::shared_ptr<dawn::sir::FunCallExpr> fcall(const std::string& callee);
std::shared_ptr<dawn::sir::StencilFunCallExpr> sfcall(const std::string& calee);
std::shared_ptr<dawn::sir::StencilFunArgExpr> arg(int direction, int offset, int argumentIndex);
std::shared_ptr<dawn::sir::VarAccessExpr> var(const std::string& name,
                                         std::shared_ptr<dawn::sir::Expr> index = nullptr);
std::shared_ptr<dawn::sir::FieldAccessExpr>
field(const std::string& name, dawn::Array3i offset = dawn::Array3i{{0, 0, 0}},
      dawn::Array3i argumentMap = dawn::Array3i{{-1, -1, -1}},
      dawn::Array3i argumentOffset = dawn::Array3i{{0, 0, 0}}, bool negateOffset = false);
std::shared_ptr<dawn::sir::LiteralAccessExpr>
lit(const std::string& value, dawn::BuiltinTypeID builtinType = dawn::BuiltinTypeID::Integer);
/// @}

} // namespace sirgen

} // namespace gtclang

#endif
