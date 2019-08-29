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

#ifndef DAWN_UNITTEST_ASTSIMPLIFIER_H
#define DAWN_UNITTEST_ASTSIMPLIFIER_H

#include "dawn/SIR/ASTExpr.h"
#include "dawn/SIR/ASTStmt.h"
#include <memory>

namespace dawn {

namespace internal {

class BlockWriter {
public:
  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<sir::Stmt>& statement, Args&&... args) {
    storage_.push_back(statement);
    recursiveBlock(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<sir::Expr>& expression, Args&&... args) {
    recursiveBlock(std::make_shared<sir::ExprStmt>(expression), std::forward<Args>(args)...);
  }

  void recursiveBlock() {}

  template <typename... Args>
  const std::vector<std::shared_ptr<sir::Stmt>>& createVec(const std::shared_ptr<sir::Stmt>& statement,
                                                      Args&&... args) {
    recursiveBlock(statement, std::forward<Args>(args)...);
    return storage_;
  }
  template <typename... Args>
  const std::vector<std::shared_ptr<sir::Stmt>>& createVec(const std::shared_ptr<sir::Expr>& expr,
                                                      Args&&... args) {
    recursiveBlock(expr, std::forward<Args>(args)...);
    return storage_;
  }

private:
  std::vector<std::shared_ptr<sir::Stmt>> storage_;
};

} // namespace internal

namespace astgen {

/// @brief Simplification for generating in-memory ASTs
///
/// THe following functions allow for a simplified notation to generate in-memory SIRs for testing
/// puropses. It can be used to describe simple operations or blocks of operations in a
/// human-readable way.
///
/// @code{.cpp}
///   assign(var("a"), binop(var("b"), "+", var("c")))
/// @endcode
/// @ingroup unittest
/// @{
template <typename... Args>
std::shared_ptr<sir::BlockStmt> block(Args&&... args) {
  internal::BlockWriter bw;
  auto vec = bw.createVec(std::forward<Args>(args)...);
  return std::make_shared<sir::BlockStmt>(vec);
}

std::shared_ptr<sir::ExprStmt> expr(const std::shared_ptr<sir::Expr>& expr);

std::shared_ptr<sir::ReturnStmt> ret(const std::shared_ptr<sir::Expr>& expr);

std::shared_ptr<sir::VarDeclStmt> vardecl(const std::string& type, const std::string& name,
                                     const std::shared_ptr<sir::Expr>& init, const char* op = "=");

std::shared_ptr<sir::VarDeclStmt> vecdecl(const std::string& type, const std::string& name,
                                     std::vector<std::shared_ptr<sir::Expr>> initList, int dimension = 0,
                                     const char* op = "=");

std::shared_ptr<sir::VerticalRegionDeclStmt>
verticalRegion(const std::shared_ptr<sir::VerticalRegion>& verticalRegion);

std::shared_ptr<sir::StencilCallDeclStmt> scdec(const std::shared_ptr<sir::StencilCall>& stencilCall);

std::shared_ptr<sir::BoundaryConditionDeclStmt> boundaryCondition(const std::string& callee);

std::shared_ptr<sir::IfStmt> ifstmt(const std::shared_ptr<sir::Stmt>& condExpr,
                               const std::shared_ptr<sir::Stmt>& thenStmt,
                               const std::shared_ptr<sir::Stmt>& elseStmt = nullptr);

std::shared_ptr<sir::UnaryOperator> unop(const std::shared_ptr<sir::Expr>& operand, const char* op);

std::shared_ptr<sir::BinaryOperator> binop(const std::shared_ptr<sir::Expr>& left, const char* op,
                                      const std::shared_ptr<sir::Expr>& right);

std::shared_ptr<sir::AssignmentExpr> assign(const std::shared_ptr<sir::Expr>& left,
                                       const std::shared_ptr<sir::Expr>& right, const char* op = "=");

std::shared_ptr<sir::TernaryOperator> ternop(const std::shared_ptr<sir::Expr>& cond,
                                        const std::shared_ptr<sir::Expr>& left,
                                        const std::shared_ptr<sir::Expr>& right);
std::shared_ptr<sir::FunCallExpr> fcall(const std::string& callee);

std::shared_ptr<sir::StencilFunCallExpr> sfcall(const std::string& callee);

std::shared_ptr<sir::StencilFunArgExpr> arg(int direction, int offset, int argumentIndex);

std::shared_ptr<sir::VarAccessExpr> var(const std::string& name, std::shared_ptr<sir::Expr> index = nullptr);
std::shared_ptr<sir::FieldAccessExpr> field(const std::string& name, Array3i offset = Array3i{{0, 0, 0}},
                                       Array3i argumentMap = Array3i{{-1, -1, -1}},
                                       Array3i argumentOffset = Array3i{{0, 0, 0}},
                                       bool negateOffset = false);

std::shared_ptr<sir::LiteralAccessExpr> lit(const std::string& value,
                                       BuiltinTypeID builtinType = BuiltinTypeID::Float);
/// @}

} // namespace astgen

} // namespace dawn

#endif
