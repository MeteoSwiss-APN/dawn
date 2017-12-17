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

#include "dawn/CodeGen/CXXNaive/ASTStencilFnParamVisitor.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilFunctionInstantiation.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace cxxnaive {

ASTCodeGenCXXNaiveStencilFnParamVisitor::ASTCodeGenCXXNaiveStencilFnParamVisitor(
    std::unordered_map<std::string, std::string> paramNameToType)
    : paramNameToType_(paramNameToType) {}

ASTCodeGenCXXNaiveStencilFnParamVisitor::~ASTCodeGenCXXNaiveStencilFnParamVisitor() {}

void ASTCodeGenCXXNaiveStencilFnParamVisitor::visit(const std::shared_ptr<VarAccessExpr>& expr) {}

void ASTCodeGenCXXNaiveStencilFnParamVisitor::visit(
    const std::shared_ptr<LiteralAccessExpr>& expr) {}

void ASTCodeGenCXXNaiveStencilFnParamVisitor::visit(const std::shared_ptr<FieldAccessExpr>& expr) {

  if(!paramNameToType_.count(expr->getName()))
    DAWN_ASSERT_MSG(0, "param of stencil function call not found");

  ss_ << "ParamWrapper<" << codegen::c_gt() << "data_view<" << paramNameToType_[expr->getName()]
      << ">>(" << expr->getName() << ","
      << "std::array<int, 3>{" << RangeToString(", ", "", "")(expr->getOffset()) << "})";
}

std::string ASTCodeGenCXXNaiveStencilFnParamVisitor::getCodeAndResetStream() {
  std::string str = ss_.str();
  codegen::clear(ss_);
  return str;
}

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
