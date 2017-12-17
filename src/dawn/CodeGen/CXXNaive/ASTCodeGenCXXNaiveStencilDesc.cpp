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

#include "dawn/CodeGen/CXXNaive/ASTCodeGenCXXNaiveStencilDesc.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace cxxnaive {

ASTCodeGenCXXNaiveStencilDesc::ASTCodeGenCXXNaiveStencilDesc(
    const dawn::StencilInstantiation* instantiation,
    std::unordered_map<int, std::vector<std::string>> const& stencilIDToStencilNameMap)
    : ASTCodeGenCXX(), instantiation_(instantiation),
      stencilIDToStencilNameMap_(stencilIDToStencilNameMap) {}

ASTCodeGenCXXNaiveStencilDesc::~ASTCodeGenCXXNaiveStencilDesc() {}

const std::string& ASTCodeGenCXXNaiveStencilDesc::getName(const std::shared_ptr<Stmt>& stmt) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

const std::string& ASTCodeGenCXXNaiveStencilDesc::getName(const std::shared_ptr<Expr>& expr) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTCodeGenCXXNaiveStencilDesc::visit(const std::shared_ptr<ReturnStmt>& stmt) {
  dawn_unreachable("ReturnStmt not allowed in StencilDesc AST");
}

void ASTCodeGenCXXNaiveStencilDesc::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  dawn_unreachable("VerticalRegionDeclStmt not allowed in StencilDesc AST");
}

void ASTCodeGenCXXNaiveStencilDesc::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  int stencilID = instantiation_->getStencilCallToStencilIDMap().find(stmt)->second;

  for(const std::string& stencilName : stencilIDToStencilNameMap_.find(stencilID)->second) {

    for(const auto& stencil : instantiation_->getStencils()) {

      if(stencil->getStencilID() != stencilID)
        continue;
      ss_ << "m_" << stencilName + "->run()";
    }
  }
}

void ASTCodeGenCXXNaiveStencilDesc::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not yet implemented");
}

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTCodeGenCXXNaiveStencilDesc::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  dawn_unreachable("StencilFunCallExpr not allowed in StencilDesc AST");
}

void ASTCodeGenCXXNaiveStencilDesc::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
  dawn_unreachable("StencilFunArgExpr not allowed in StencilDesc AST");
}

void ASTCodeGenCXXNaiveStencilDesc::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  if(instantiation_->isGlobalVariable(instantiation_->getAccessIDFromExpr(expr)))
    ss_ << "globals::get().";

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

void ASTCodeGenCXXNaiveStencilDesc::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  dawn_unreachable("FieldAccessExpr not allowed in StencilDesc AST");
}

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
