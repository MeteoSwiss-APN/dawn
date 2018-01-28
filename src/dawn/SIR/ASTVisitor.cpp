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

#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"

namespace dawn {

ASTVisitor::~ASTVisitor() {}

#define ASTVISITORFORWARDING_VISIT_IMPL(Type)                                                      \
  void ASTVisitorForwarding::visit(const std::shared_ptr<Type>& node) {                            \
    for(const auto& s : node->getChildren())                                                       \
      s->accept(*this);                                                                            \
  }

ASTVISITORFORWARDING_VISIT_IMPL(BlockStmt);
ASTVISITORFORWARDING_VISIT_IMPL(StencilCallDeclStmt);
ASTVISITORFORWARDING_VISIT_IMPL(BoundaryConditionDeclStmt);
ASTVISITORFORWARDING_VISIT_IMPL(IfStmt);
ASTVISITORFORWARDING_VISIT_IMPL(UnaryOperator);
ASTVISITORFORWARDING_VISIT_IMPL(BinaryOperator);
ASTVISITORFORWARDING_VISIT_IMPL(AssignmentExpr);
ASTVISITORFORWARDING_VISIT_IMPL(TernaryOperator);
ASTVISITORFORWARDING_VISIT_IMPL(FunCallExpr);
ASTVISITORFORWARDING_VISIT_IMPL(StencilFunCallExpr);
ASTVISITORFORWARDING_VISIT_IMPL(StencilFunArgExpr);
ASTVISITORFORWARDING_VISIT_IMPL(VarAccessExpr);
ASTVISITORFORWARDING_VISIT_IMPL(FieldAccessExpr);
ASTVISITORFORWARDING_VISIT_IMPL(LiteralAccessExpr);

#undef ASTVISITORFORWARDING_VISIT_IMPL

#define ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(Type)                                              \
  void ASTVisitorForwardingNonConst::visit(std::shared_ptr<Type> node) {                           \
    for(auto& s : node->getChildren())                                                             \
      s->accept(*this);                                                                            \
  }

ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(BlockStmt);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(StencilCallDeclStmt);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(BoundaryConditionDeclStmt);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(IfStmt);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(UnaryOperator);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(BinaryOperator);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(AssignmentExpr);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(TernaryOperator);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(FunCallExpr);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(StencilFunCallExpr);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(StencilFunArgExpr);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(VarAccessExpr);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(FieldAccessExpr);
ASTVISITORFORWARDINGNONCONST_VISIT_IMPL(LiteralAccessExpr);

#undef ASTVISITORFORWARDINGNONCONST_VISIT_IMPL

#define ASTVISITORDISABLED_VISIT_IMPL(Type)                                                        \
  void ASTVisitorDisabled::visit(const std::shared_ptr<Type>& node) {                              \
    DAWN_ASSERT_MSG(0, "Type not allowed in this context");                                        \
  }

ASTVISITORDISABLED_VISIT_IMPL(BlockStmt);
ASTVISITORDISABLED_VISIT_IMPL(StencilCallDeclStmt);
ASTVISITORDISABLED_VISIT_IMPL(BoundaryConditionDeclStmt);
ASTVISITORDISABLED_VISIT_IMPL(IfStmt);
ASTVISITORDISABLED_VISIT_IMPL(UnaryOperator);
ASTVISITORDISABLED_VISIT_IMPL(BinaryOperator);
ASTVISITORDISABLED_VISIT_IMPL(AssignmentExpr);
ASTVISITORDISABLED_VISIT_IMPL(TernaryOperator);
ASTVISITORDISABLED_VISIT_IMPL(FunCallExpr);
ASTVISITORDISABLED_VISIT_IMPL(StencilFunCallExpr);
ASTVISITORDISABLED_VISIT_IMPL(StencilFunArgExpr);
ASTVISITORDISABLED_VISIT_IMPL(VarAccessExpr);
ASTVISITORDISABLED_VISIT_IMPL(FieldAccessExpr);
ASTVISITORDISABLED_VISIT_IMPL(LiteralAccessExpr);
ASTVISITORDISABLED_VISIT_IMPL(ExprStmt);
ASTVISITORDISABLED_VISIT_IMPL(ReturnStmt);
ASTVISITORDISABLED_VISIT_IMPL(VarDeclStmt);
ASTVISITORDISABLED_VISIT_IMPL(VerticalRegionDeclStmt);

#undef ASTVISITORDISABLED_VISIT_IMPL

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

void ASTVisitorForwarding::visit(const std::shared_ptr<dawn::VerticalRegionDeclStmt>& stmt) {
  stmt->getVerticalRegion()->Ast->accept(*this);
}

ASTVisitorForwarding::~ASTVisitorForwarding() {}
ASTVisitorDisabled::~ASTVisitorDisabled() {}

} // namespace dawn
