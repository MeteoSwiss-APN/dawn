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

#include "dawn/AST/ASTVisitor.h"
#include "dawn/AST/AST.h"
#include "dawn/SIR/SIR.h"

namespace dawn {
namespace ast {
ASTVisitor::~ASTVisitor() {}
ASTVisitorNonConst::~ASTVisitorNonConst() {}
ASTVisitorForwarding::~ASTVisitorForwarding() {}
ASTVisitorForwardingNonConst::~ASTVisitorForwardingNonConst() {}
ASTVisitorDisabled::~ASTVisitorDisabled() {}
ASTVisitorPostOrder::~ASTVisitorPostOrder() {}

#define ASTVISITORFORWARDING_VISIT_IMPL(Type)                                                      \
  void ASTVisitorForwarding::visit(const std::shared_ptr<Type>& node) {                            \
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
ASTVISITORFORWARDING_VISIT_IMPL(ReductionOverNeighborStmt)

#undef ASTVISITORFORWARDING_VISIT_IMPL

void ASTVisitorForwarding::visit(const std::shared_ptr<ExprStmt>& node) {
  node->getExpr()->accept(*this);
}

void ASTVisitorForwarding::visit(const std::shared_ptr<ReturnStmt>& node) {
  node->getExpr()->accept(*this);
}

void ASTVisitorForwarding::visit(const std::shared_ptr<VarDeclStmt>& node) {
  for(const auto& expr : node->getInitList())
    expr->accept(*this);
}

void ASTVisitorForwarding::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  stmt->getVerticalRegion()->Ast->accept(*this);
}

#define ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(Type)                                              \
  void ASTVisitorForwardingNonConst::visit(std::shared_ptr<Type> node) {                           \
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
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(ReductionOverNeighborStmt)

#undef ASTVISITORFORWARDINGNONCONST_VISIT_IMPL

void ASTVisitorForwardingNonConst::visit(std::shared_ptr<ExprStmt> node) {
  node->getExpr()->accept(*this);
}

void ASTVisitorForwardingNonConst::visit(std::shared_ptr<ReturnStmt> node) {
  node->getExpr()->accept(*this);
}

void ASTVisitorForwardingNonConst::visit(std::shared_ptr<VarDeclStmt> node) {
  for(const auto& expr : node->getInitList())
    expr->accept(*this);
}

void ASTVisitorForwardingNonConst::visit(std::shared_ptr<VerticalRegionDeclStmt> stmt) {
  stmt->getVerticalRegion()->Ast->accept(*this);
}

#define ASTVISITORPOSTORDER_VISIT_IMPL(NodeType, Type)                                             \
  std::shared_ptr<NodeType> ASTVisitorPostOrder::visitAndReplace(                                  \
      std::shared_ptr<Type> const& node) {                                                         \
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
  bool ASTVisitorPostOrder::preVisitNode(std::shared_ptr<Type> const& node) { return true; }       \
  std::shared_ptr<NodeType> ASTVisitorPostOrder::postVisitNode(                                    \
      std::shared_ptr<Type> const& node) {                                                         \
    return node;                                                                                   \
  }

ASTVISITORPOSTORDER_VISIT_IMPL(Stmt, BlockStmt)
ASTVISITORPOSTORDER_VISIT_IMPL(Stmt, StencilCallDeclStmt)
ASTVISITORPOSTORDER_VISIT_IMPL(Stmt, BoundaryConditionDeclStmt)
ASTVISITORPOSTORDER_VISIT_IMPL(Stmt, IfStmt)
ASTVISITORPOSTORDER_VISIT_IMPL(Stmt, ReductionOverNeighborStmt)
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

std::shared_ptr<Stmt> ASTVisitorPostOrder::visitAndReplace(std::shared_ptr<ExprStmt> const& node) {
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
bool ASTVisitorPostOrder::preVisitNode(std::shared_ptr<ExprStmt> const& node) { return true; }
std::shared_ptr<Stmt> ASTVisitorPostOrder::postVisitNode(std::shared_ptr<ExprStmt> const& node) {
  return node;
}

std::shared_ptr<Stmt>
ASTVisitorPostOrder::visitAndReplace(std::shared_ptr<ReturnStmt> const& node) {
  if(!preVisitNode(node))
    return node;
  auto repl = node->getExpr()->acceptAndReplace(*this);
  if(repl && repl != node->getExpr())
    node->replaceChildren(node->getExpr(), repl);
  return postVisitNode(node);
}
std::shared_ptr<Stmt> ASTVisitorPostOrder::postVisitNode(std::shared_ptr<ReturnStmt> const& node) {
  return node;
}
bool ASTVisitorPostOrder::preVisitNode(std::shared_ptr<ReturnStmt> const& node) { return true; }

std::shared_ptr<Stmt>
ASTVisitorPostOrder::visitAndReplace(std::shared_ptr<VarDeclStmt> const& node) {
  if(!preVisitNode(node))
    return node;
  for(auto expr : node->getInitList()) {
    auto repl = expr->acceptAndReplace(*this);
    if(repl && repl != expr)
      node->replaceChildren(expr, repl);
  }
  return postVisitNode(node);
}

std::shared_ptr<Stmt> ASTVisitorPostOrder::postVisitNode(std::shared_ptr<VarDeclStmt> const& node) {
  return node;
}
bool ASTVisitorPostOrder::preVisitNode(std::shared_ptr<VarDeclStmt> const& node) { return true; }

std::shared_ptr<Stmt>
ASTVisitorPostOrder::visitAndReplace(std::shared_ptr<VerticalRegionDeclStmt> const& stmt) {
  // TODO replace this as wel
  if(!preVisitNode(stmt))
    return stmt;
  auto repl = stmt->getVerticalRegion()->Ast->acceptAndReplace(*this);
  if(repl && repl != stmt->getVerticalRegion()->Ast)
    stmt->getVerticalRegion()->Ast = repl;
  return postVisitNode(stmt);
}

bool ASTVisitorPostOrder::preVisitNode(std::shared_ptr<VerticalRegionDeclStmt> const& stmt) {
  return true;
}

std::shared_ptr<Stmt>
ASTVisitorPostOrder::postVisitNode(std::shared_ptr<VerticalRegionDeclStmt> const& stmt) {
  return stmt;
}

#define ASTVISITORDISABLED_VISIT_IMPL(Type)                                                        \
  void ASTVisitorDisabled::visit(const std::shared_ptr<Type>& node) {                              \
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
ASTVISITORDISABLED_VISIT_IMPL(VerticalRegionDeclStmt)
ASTVISITORDISABLED_VISIT_IMPL(ReductionOverNeighborStmt)

#undef ASTVISITORDISABLED_VISIT_IMPL
} // namespace ast
} // namespace dawn
