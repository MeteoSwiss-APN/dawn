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

#include "dawn/CodeGen/ASTCodeGenGTClangStencilDesc.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {

ASTCodeGenGTClangStencilDesc::ASTCodeGenGTClangStencilDesc(
    const StencilInstantiation* instantiation,
    const std::unordered_map<int, std::vector<std::string>>& StencilIDToStencilNameMap)
    : ASTCodeGenCXX(), instantiation_(instantiation),
      StencilIDToStencilNameMap_(StencilIDToStencilNameMap) {}

ASTCodeGenGTClangStencilDesc::~ASTCodeGenGTClangStencilDesc() {}

const std::string& ASTCodeGenGTClangStencilDesc::getName(const std::shared_ptr<Stmt>& stmt) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

const std::string& ASTCodeGenGTClangStencilDesc::getName(const std::shared_ptr<Expr>& expr) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<BlockStmt>& stmt) {
  Base::visit(stmt);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<ExprStmt>& stmt) {
  Base::visit(stmt);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<ReturnStmt>& stmt) {
  dawn_unreachable("ReturnStmt not allowed in StencilDesc AST");
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<VarDeclStmt>& stmt) {
  Base::visit(stmt);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  dawn_unreachable("VerticalRegionDeclStmt not allowed in StencilDesc AST");
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  int StencilID = instantiation_->getStencilCallToStencilIDMap().find(stmt)->second;
  for(const std::string& stencilName : StencilIDToStencilNameMap_.find(StencilID)->second)
    ss_ << std::string(indent_, ' ') << stencilName << ".get_stencil()->run();\n";
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
    std::cout << "boundary condition not implemented in ASTCodeGenGTClangStencilDesc" << std::endl;
    //  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not yet implemented");
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<IfStmt>& stmt) { Base::visit(stmt); }

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<UnaryOperator>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<BinaryOperator>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<AssignmentExpr>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<TernaryOperator>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<FunCallExpr>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  dawn_unreachable("StencilFunCallExpr not allowed in StencilDesc AST");
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
  dawn_unreachable("StencilFunArgExpr not allowed in StencilDesc AST");
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  if(instantiation_->isGlobalVariable(instantiation_->getAccessIDFromExpr(expr)))
    ss_ << "globals::get().";

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<LiteralAccessExpr>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  dawn_unreachable("FieldAccessExpr not allowed in StencilDesc AST");
}

} // namespace dawn
