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

#include "dawn/Unittest/PrintAllExpressionTypes.h"
#include "dawn/SIR/ASTStringifier.h"
#include "dawn/SIR/SIR.h"
#include <iostream>

namespace dawn {

void PrintAllExpressionTypes::visit(const std::shared_ptr<BlockStmt>& node) {
  std::cout << "Block Statement\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<VerticalRegionDeclStmt>& node) {
  std::cout << "VerticalRegionDeclStmt\n" << ASTStringifer::toString(node) << std::endl;
  node->getVerticalRegion()->Ast->accept(*this);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<StencilCallDeclStmt>& node) {
  std::cout << "StencilCallDeclStmt\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& node) {
  std::cout << "BoundaryConditionDeclStmt\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<IfStmt>& node) {
  std::cout << "IfStmt\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<UnaryOperator>& node) {
  std::cout << "UnaryOperator\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<BinaryOperator>& node) {
  std::cout << "BinaryOperator\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<AssignmentExpr>& node) {
  std::cout << "AssignmentExpr\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<TernaryOperator>& node) {
  std::cout << "TernaryOperator\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<FunCallExpr>& node) {
  std::cout << "FunCallExpr\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<StencilFunCallExpr>& node) {
  std::cout << "StencilFunCallExpr\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<StencilFunArgExpr>& node) {
  std::cout << "StencilFunArgExpr\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<VarAccessExpr>& node) {
  std::cout << "VarAccessExpr\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<FieldAccessExpr>& node) {
  std::cout << "FieldAccessExpr\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<LiteralAccessExpr>& node) {
  std::cout << "LiteralAccessExpr\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<ExprStmt>& node) {
  std::cout << "ExprStmt\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<ReturnStmt>& node) {
  std::cout << "ReturnStmt\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<VarDeclStmt>& node) {
  std::cout << "VarDeclStmt\n" << ASTStringifer::toString(node) << std::endl;
  ASTVisitorForwarding::visit(node);
}

} // namespace dawn
