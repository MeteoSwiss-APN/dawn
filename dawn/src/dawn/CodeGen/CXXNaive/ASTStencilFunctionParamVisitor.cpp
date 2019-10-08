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

#include "dawn/CodeGen/CXXNaive/ASTStencilFunctionParamVisitor.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace cxxnaive {

ASTStencilFunctionParamVisitor::ASTStencilFunctionParamVisitor(
    const std::shared_ptr<iir::StencilFunctionInstantiation>& function,
    const iir::StencilMetaInformation& metadata)
    : metadata_(metadata), currentFunction_(function) {}

ASTStencilFunctionParamVisitor::~ASTStencilFunctionParamVisitor() {}

std::string ASTStencilFunctionParamVisitor::getName(const std::shared_ptr<iir::Expr>& expr) const {

  if(currentFunction_)
    return currentFunction_->getFieldNameFromAccessID(getAccessID(expr));
  else
    return metadata_.getFieldNameFromAccessID(getAccessID(expr));
}

int ASTStencilFunctionParamVisitor::getAccessID(const std::shared_ptr<iir::Expr>& expr) const {
  if(currentFunction_)
    return currentFunction_->getAccessIDFromExpr(expr);
  else
    return metadata_.getAccessIDFromExpr(expr);
}

void ASTStencilFunctionParamVisitor::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {}

void ASTStencilFunctionParamVisitor::visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {}

void ASTStencilFunctionParamVisitor::visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) {}

void ASTStencilFunctionParamVisitor::visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {

  for(auto& arg : expr->getArguments()) {
    arg->accept(*this);
  }
}

void ASTStencilFunctionParamVisitor::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {

  std::string fieldName = (currentFunction_) ? currentFunction_->getOriginalNameFromCallerAccessID(
                                                   currentFunction_->getAccessIDFromExpr(expr))
                                             : getName(expr);

  ss_ << ",param_wrapper<decltype(" << fieldName << ")>(" << fieldName << ","
      << "std::array<int, 3>{" << RangeToString(", ", "", "")(expr->getOffset())
      << "}+" + fieldName + "_offsets)";
}

std::string ASTStencilFunctionParamVisitor::getCodeAndResetStream() {
  std::string str = ss_.str();
  codegen::clear(ss_);
  return str;
}

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
