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

#include <string>
#include "dawn/CodeGen/Cuda/ASTStencilBody.h"
#include "dawn/CodeGen/Cuda/ASTStencilFunctionParamVisitor.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace cuda {

ASTStencilBody::ASTStencilBody(const iir::StencilInstantiation* stencilInstantiation,
                               StencilContext stencilContext)
    : ASTCodeGenCXX(), instantiation_(stencilInstantiation), offsetPrinter_("+", "", "", true) {}

ASTStencilBody::~ASTStencilBody() {}

std::string ASTStencilBody::getName(const std::shared_ptr<Expr>& expr) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

std::string ASTStencilBody::getName(const std::shared_ptr<Stmt>& stmt) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<ReturnStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTStencilBody::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  dawn_unreachable("stencil functions not allows in cuda backend");
}

void ASTStencilBody::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
  dawn_unreachable("stencil functions not allows in cuda backend");
}

void ASTStencilBody::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int AccessID = instantiation_->getAccessIDFromExpr(expr);

  if(instantiation_->isGlobalVariable(AccessID)) {
    ss_ << "globals_." << name;
  } else {
    ss_ << name;

    if(expr->isArrayAccess()) {
      ss_ << "[";
      expr->getIndex()->accept(*this);
      ss_ << "]";
    }
  }
}

void ASTStencilBody::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  std::string accessName = getName(expr);
  bool isTemporary = instantiation_->isTemporaryField(instantiation_->getAccessIDFromExpr(expr));
  std::string index = isTemporary ? "idx_tmp" : "idx";

  std::string offsetStr = offsetPrinter_(ijkfyOffset(expr->getOffset(), accessName, isTemporary));
  ss_ << accessName
      << (offsetStr.empty() ? "[" + index + "]" : ("[" + index + "+" + offsetStr + "]"));
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
