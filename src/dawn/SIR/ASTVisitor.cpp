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

ASTVisitor::~ASTVisitor() {}

} // namespace dawn
