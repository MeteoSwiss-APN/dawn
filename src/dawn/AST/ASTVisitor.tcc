//===--------------------------------------------------------------------------------*-
// C++ -*-===//
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

#include "dawn/AST/AST.h"
#include "dawn/Support/Assert.h"

namespace dawn {
namespace ast {
template <typename DataTraits>
ASTVisitor<DataTraits>::~ASTVisitor() {}
template <typename DataTraits>
ASTVisitorNonConst<DataTraits>::~ASTVisitorNonConst() {}
template <typename DataTraits>
ASTVisitorForwarding<DataTraits>::~ASTVisitorForwarding() {}
template <typename DataTraits>
ASTVisitorForwardingNonConst<DataTraits>::~ASTVisitorForwardingNonConst() {}
template <typename DataTraits>
ASTVisitorDisabled<DataTraits>::~ASTVisitorDisabled() {}
template <typename DataTraits>
ASTVisitorPostOrder<DataTraits>::~ASTVisitorPostOrder() {}

#define ASTVISITORFORWARDING_VISIT_IMPL(Type)                                                      \
  template <typename DataTraits>                                                                   \
  void ASTVisitorForwarding<DataTraits>::visit(const std::shared_ptr<Type<DataTraits>>& node) {    \
    for(const auto& s : node->getChildren())                                                       \
      s->accept(*this);                                                                            \
  }

ASTVISITORFORWARDING_VISIT_IMPL(BlockStmt)
ASTVISITORFORWARDING_VISIT_IMPL(StencilCallDeclStmt)
ASTVISITORFORWARDING_VISIT_IMPL(BoundaryConditionDeclStmt)
ASTVISITORFORWARDING_VISIT_IMPL(IfStmt)
ASTVISITORFORWARDING_VISIT_IMPL(UnaryOperator)
ASTVISITORFORWARDING_VISIT_IMPL(BinaryOperator)
ASTVISITORFORWARDING_VISIT_IMPL(AssignmentExpr)
ASTVISITORFORWARDING_VISIT_IMPL(TernaryOperator)
ASTVISITORFORWARDING_VISIT_IMPL(FunCallExpr)
ASTVISITORFORWARDING_VISIT_IMPL(StencilFunCallExpr)
ASTVISITORFORWARDING_VISIT_IMPL(StencilFunArgExpr)
ASTVISITORFORWARDING_VISIT_IMPL(VarAccessExpr)
ASTVISITORFORWARDING_VISIT_IMPL(FieldAccessExpr)
ASTVISITORFORWARDING_VISIT_IMPL(LiteralAccessExpr)

#undef ASTVISITORFORWARDING_VISIT_IMPL

template <typename DataTraits>
void ASTVisitorForwarding<DataTraits>::visit(const std::shared_ptr<ExprStmt<DataTraits>>& node) {
  node->getExpr()->accept(*this);
}

template <typename DataTraits>
void ASTVisitorForwarding<DataTraits>::visit(const std::shared_ptr<ReturnStmt<DataTraits>>& node) {
  node->getExpr()->accept(*this);
}

template <typename DataTraits>
void ASTVisitorForwarding<DataTraits>::visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& node) {
  for(const auto& expr : node->getInitList())
    expr->accept(*this);
}

#define ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(Type)                                              \
  template <typename DataTraits>                                                                   \
  void ASTVisitorForwardingNonConst<DataTraits>::visit(std::shared_ptr<Type<DataTraits>> node) {   \
    for(auto& s : node->getChildren())                                                             \
      s->accept(*this);                                                                            \
  }

ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(BlockStmt)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(StencilCallDeclStmt)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(BoundaryConditionDeclStmt)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(IfStmt)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(UnaryOperator)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(BinaryOperator)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(AssignmentExpr)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(TernaryOperator)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(FunCallExpr)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(StencilFunCallExpr)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(StencilFunArgExpr)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(VarAccessExpr)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(FieldAccessExpr)
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(LiteralAccessExpr)

#undef ASTVISITORFORWARDINGNONCONST_VISIT_IMPL

template <typename DataTraits>
void ASTVisitorForwardingNonConst<DataTraits>::visit(std::shared_ptr<ExprStmt<DataTraits>> node) {
  node->getExpr()->accept(*this);
}

template <typename DataTraits>
void ASTVisitorForwardingNonConst<DataTraits>::visit(std::shared_ptr<ReturnStmt<DataTraits>> node) {
  node->getExpr()->accept(*this);
}

template <typename DataTraits>
void ASTVisitorForwardingNonConst<DataTraits>::visit(
    std::shared_ptr<VarDeclStmt<DataTraits>> node) {
  for(const auto& expr : node->getInitList())
    expr->accept(*this);
}

#define ASTVISITORPOSTORDER_VISIT_IMPL(NodeType, Type)                                             \
  template <typename DataTraits>                                                                   \
  std::shared_ptr<NodeType<DataTraits>> ASTVisitorPostOrder<DataTraits>::visitAndReplace(          \
      std::shared_ptr<Type<DataTraits>> const& node) {                                             \
    if(!preVisitNode(node))                                                                        \
      return node;                                                                                 \
    for(auto s : node->getChildren()) {                                                            \
      auto repl = s->acceptAndReplace(*this);                                                      \
      if(repl && repl != s) {                                                                      \
        node->replaceChildren(s, repl);                                                            \
      }                                                                                            \
    }                                                                                              \
    return postVisitNode(node);                                                                    \
  }                                                                                                \
  template <typename DataTraits>                                                                   \
  bool ASTVisitorPostOrder<DataTraits>::preVisitNode(                                              \
      std::shared_ptr<Type<DataTraits>> const& node) {                                             \
    return true;                                                                                   \
  }                                                                                                \
  template <typename DataTraits>                                                                   \
  std::shared_ptr<NodeType<DataTraits>> ASTVisitorPostOrder<DataTraits>::postVisitNode(            \
      std::shared_ptr<Type<DataTraits>> const& node) {                                             \
    return node;                                                                                   \
  }

ASTVISITORPOSTORDER_VISIT_IMPL(Stmt, BlockStmt)
ASTVISITORPOSTORDER_VISIT_IMPL(Stmt, StencilCallDeclStmt)
ASTVISITORPOSTORDER_VISIT_IMPL(Stmt, BoundaryConditionDeclStmt)
ASTVISITORPOSTORDER_VISIT_IMPL(Stmt, IfStmt)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, UnaryOperator)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, BinaryOperator)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, AssignmentExpr)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, TernaryOperator)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, FunCallExpr)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, StencilFunCallExpr)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, StencilFunArgExpr)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, VarAccessExpr)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, FieldAccessExpr)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, LiteralAccessExpr)
ASTVISITORPOSTORDER_VISIT_IMPL(Expr, NOPExpr)

#undef ASTVISITORPOSTORDER_VISIT_STMT_IMPL
#undef ASTVISITORPOSTORDER_VISIT_EXPR_IMPL

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> ASTVisitorPostOrder<DataTraits>::visitAndReplace(
    std::shared_ptr<ExprStmt<DataTraits>> const& node) {
  DAWN_ASSERT(node);
  if(!preVisitNode(node))
    return node;

  DAWN_ASSERT(node->getExpr());
  auto repl = node->getExpr()->acceptAndReplace(*this);
  DAWN_ASSERT(repl);

  if(repl && repl != node->getExpr()) {
    auto oo = node->getExpr();
    DAWN_ASSERT(oo && repl);
    node->replaceChildren(node->getExpr(), repl);
  }
  return postVisitNode(node);
}
template <typename DataTraits>
bool ASTVisitorPostOrder<DataTraits>::preVisitNode(
    std::shared_ptr<ExprStmt<DataTraits>> const& node) {
  return true;
}
template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>>
ASTVisitorPostOrder<DataTraits>::postVisitNode(std::shared_ptr<ExprStmt<DataTraits>> const& node) {
  return node;
}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> ASTVisitorPostOrder<DataTraits>::visitAndReplace(
    std::shared_ptr<ReturnStmt<DataTraits>> const& node) {
  if(!preVisitNode(node))
    return node;
  auto repl = node->getExpr()->acceptAndReplace(*this);
  if(repl && repl != node->getExpr())
    node->replaceChildren(node->getExpr(), repl);
  return postVisitNode(node);
}
template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> ASTVisitorPostOrder<DataTraits>::postVisitNode(
    std::shared_ptr<ReturnStmt<DataTraits>> const& node) {
  return node;
}
template <typename DataTraits>
bool ASTVisitorPostOrder<DataTraits>::preVisitNode(
    std::shared_ptr<ReturnStmt<DataTraits>> const& node) {
  return true;
}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> ASTVisitorPostOrder<DataTraits>::visitAndReplace(
    std::shared_ptr<VarDeclStmt<DataTraits>> const& node) {
  if(!preVisitNode(node))
    return node;
  for(auto expr : node->getInitList()) {
    auto repl = expr->acceptAndReplace(*this);
    if(repl && repl != expr)
      node->replaceChildren(expr, repl);
  }
  return postVisitNode(node);
}

template <typename DataTraits>
std::shared_ptr<Stmt<DataTraits>> ASTVisitorPostOrder<DataTraits>::postVisitNode(
    std::shared_ptr<VarDeclStmt<DataTraits>> const& node) {
  return node;
}
template <typename DataTraits>
bool ASTVisitorPostOrder<DataTraits>::preVisitNode(
    std::shared_ptr<VarDeclStmt<DataTraits>> const& node) {
  return true;
}

#define ASTVISITORDISABLED_VISIT_IMPL(Type)                                                        \
  template <typename DataTraits>                                                                   \
  void ASTVisitorDisabled<DataTraits>::visit(const std::shared_ptr<Type<DataTraits>>& node) {      \
    DAWN_ASSERT_MSG(0, "Type not allowed in this context");                                        \
  }

ASTVISITORDISABLED_VISIT_IMPL(BlockStmt)
ASTVISITORDISABLED_VISIT_IMPL(StencilCallDeclStmt)
ASTVISITORDISABLED_VISIT_IMPL(BoundaryConditionDeclStmt)
ASTVISITORDISABLED_VISIT_IMPL(IfStmt)
ASTVISITORDISABLED_VISIT_IMPL(UnaryOperator)
ASTVISITORDISABLED_VISIT_IMPL(BinaryOperator)
ASTVISITORDISABLED_VISIT_IMPL(AssignmentExpr)
ASTVISITORDISABLED_VISIT_IMPL(TernaryOperator)
ASTVISITORDISABLED_VISIT_IMPL(FunCallExpr)
ASTVISITORDISABLED_VISIT_IMPL(StencilFunCallExpr)
ASTVISITORDISABLED_VISIT_IMPL(StencilFunArgExpr)
ASTVISITORDISABLED_VISIT_IMPL(VarAccessExpr)
ASTVISITORDISABLED_VISIT_IMPL(FieldAccessExpr)
ASTVISITORDISABLED_VISIT_IMPL(LiteralAccessExpr)
ASTVISITORDISABLED_VISIT_IMPL(ExprStmt)
ASTVISITORDISABLED_VISIT_IMPL(ReturnStmt)
ASTVISITORDISABLED_VISIT_IMPL(VarDeclStmt)

#undef ASTVISITORDISABLED_VISIT_IMPL
} // namespace ast
} // namespace dawn
