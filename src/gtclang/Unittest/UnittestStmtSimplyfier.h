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

class BlockWriter {
public:
  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<dawn::Stmt>& statement, Args&&... args) {
    storage_.push_back(statement);
    recursiveBlock(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<dawn::Expr>& expression, Args&&... args) {
    recursiveBlock(std::make_shared<dawn::ExprStmt>(expression), std::forward<Args>(args)...);
  }

  void recursiveBlock() {}

  template <typename... Args>
  const std::vector<std::shared_ptr<dawn::Stmt>>&
  createVec(const std::shared_ptr<dawn::Stmt>& statement, Args&&... args) {
    recursiveBlock(statement, std::forward<Args>(args)...);
    return storage_;
  }
  template <typename... Args>
  const std::vector<std::shared_ptr<dawn::Stmt>>& createVec(const std::shared_ptr<dawn::Expr>& expr,
                                                            Args&&... args) {
    recursiveBlock(expr, std::forward<Args>(args)...);
    return storage_;
  }

private:
  std::vector<std::shared_ptr<dawn::Stmt>> storage_;
};

/// @brief simplification for generating SIR in memory
/// This group of statements allows for a simplyfied notation to generate in-memory SIRs for testing
/// puropses. It can be used to describe simple operations or blocks of operations in a human
/// readable way like
/// \code{.cpp}
/// assign(var("a"), binop(var("b"),"+",var("c")))
/// \endcode
/// @ingroup unittest
/// @{
template <typename... Args>
std::shared_ptr<dawn::BlockStmt> block(Args&&... args) {
  BlockWriter bw;
  auto vec = bw.createVec(std::forward<Args>(args)...);
  return std::make_shared<dawn::BlockStmt>(vec);
}

std::shared_ptr<dawn::ExprStmt> expr(const std::shared_ptr<dawn::Expr>& expr);
std::shared_ptr<dawn::ReturnStmt> ret(const std::shared_ptr<dawn::Expr>& expr);
std::shared_ptr<dawn::VarDeclStmt> vardec(const std::string& type, const std::string& name,
                                          const std::shared_ptr<dawn::Expr>& init,
                                          const char* op = "=");
std::shared_ptr<dawn::VarDeclStmt> vecdec(const std::string& type, const std::string& name,
                                          std::vector<std::shared_ptr<dawn::Expr>> initList,
                                          int dimension = 0, const char* op = "=");
std::shared_ptr<dawn::VerticalRegionDeclStmt>
vrdec(const std::shared_ptr<dawn::sir::VerticalRegion>& verticalRegion);
std::shared_ptr<dawn::StencilCallDeclStmt>
scdec(const std::shared_ptr<dawn::sir::StencilCall>& stencilCall);
std::shared_ptr<dawn::BoundaryConditionDeclStmt> bcdec(const std::string& callee);
std::shared_ptr<dawn::IfStmt> ifst(const std::shared_ptr<dawn::Stmt>& condExpr,
                                   const std::shared_ptr<dawn::Stmt>& thenStmt,
                                   const std::shared_ptr<dawn::Stmt>& elseStmt = nullptr);
std::shared_ptr<dawn::UnaryOperator> unop(const std::shared_ptr<dawn::Expr>& operand,
                                          const char* op);
std::shared_ptr<dawn::BinaryOperator> binop(const std::shared_ptr<dawn::Expr>& left, const char* op,
                                            const std::shared_ptr<dawn::Expr>& right);
std::shared_ptr<dawn::AssignmentExpr> assign(const std::shared_ptr<dawn::Expr>& left,
                                             const std::shared_ptr<dawn::Expr>& right,
                                             const char* op = "=");
std::shared_ptr<dawn::TernaryOperator> ternop(const std::shared_ptr<dawn::Expr>& cond,
                                              const std::shared_ptr<dawn::Expr>& left,
                                              const std::shared_ptr<dawn::Expr>& right);
std::shared_ptr<dawn::FunCallExpr> fcall(const std::string& callee);
std::shared_ptr<dawn::StencilFunCallExpr> sfcall(const std::string& calee);
std::shared_ptr<dawn::StencilFunArgExpr> sfarg(int direction, int offset, int argumentIndex);
std::shared_ptr<dawn::VarAccessExpr> var(const std::string& name,
                                         std::shared_ptr<dawn::Expr> index = nullptr);
std::shared_ptr<dawn::FieldAccessExpr>
field(const std::string& name, dawn::Array3i offset = dawn::Array3i{{0, 0, 0}},
      dawn::Array3i argumentMap = dawn::Array3i{{-1, -1, -1}},
      dawn::Array3i argumentOffset = dawn::Array3i{{0, 0, 0}}, bool negateOffset = false);
std::shared_ptr<dawn::LiteralAccessExpr>
lit(const std::string& value, dawn::BuiltinTypeID builtinType = dawn::BuiltinTypeID::Integer);
/// @}

} // namespace sirgen

} // namespace gtclang

#endif
