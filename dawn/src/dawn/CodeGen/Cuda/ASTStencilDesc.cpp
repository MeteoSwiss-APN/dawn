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

#include "dawn/CodeGen/Cuda/ASTStencilDesc.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/StencilFunctionAsBCGenerator.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/Support/IndexRange.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace cuda {

ASTStencilDesc::ASTStencilDesc(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                               CodeGenProperties const& codeGenProperties)
    : ASTCodeGenCXX(), instantiation_(instantiation), metadata_(instantiation->getMetaData()),
      codeGenProperties_(codeGenProperties) {}

ASTStencilDesc::~ASTStencilDesc() {}

std::string ASTStencilDesc::getName(const std::shared_ptr<iir::VarDeclStmt>& stmt) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessIDFromStmt(stmt));
}

std::string ASTStencilDesc::getName(const std::shared_ptr<iir::Expr>& expr) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessIDFromExpr(expr));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilDesc::visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "ReturnStmt not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {
  int stencilID = metadata_.getStencilIDFromStencilCallStmt(stmt);

  const iir::Stencil& stencil = instantiation_->getIIR()->getStencil(stencilID);

  // fields used in the stencil
  const auto stencilFields = stencil.getOrderedFields();

  auto nonTempFields = makeRange(
      stencilFields,
      std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
          [](std::pair<int, iir::Stencil::FieldInfo> const& p) { return !p.second.IsTemporary; }));

  std::string stencilName =
      codeGenProperties_.getStencilName(StencilContext::SC_Stencil, stencilID);
  ss_ << "m_" << stencilName + "->run";

  RangeToString fieldArgs(",", "(", ");");

  ss_ << fieldArgs(nonTempFields, [&](const std::pair<const int, iir::Stencil::FieldInfo>& fieldp) {
    if(metadata_.isAccessType(iir::FieldAccessType::FAT_InterStencilTemporary, fieldp.first)) {
      return "m_" + fieldp.second.Name;
    } else {
      return fieldp.second.Name;
    }
  });

  ss_ << std::endl;
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
  BCGenerator bcGen(metadata_, ss_);
  bcGen.generate(stmt);
}

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilDesc::visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
  DAWN_ASSERT_MSG(0, "StencilFunCallExpr not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {
  DAWN_ASSERT_MSG(0, "StencilFunArgExpr not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {
  if(metadata_.isAccessType(iir::FieldAccessType::FAT_GlobalVariable,
                            iir::getAccessIDFromExpr(expr))) {
    ss_ << "m_globals.";
  }

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  DAWN_ASSERT_MSG(0, "FieldAccessExpr not allowed in StencilDesc AST");
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
