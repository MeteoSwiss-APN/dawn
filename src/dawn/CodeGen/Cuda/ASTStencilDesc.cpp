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
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace cuda {

ASTStencilDesc::ASTStencilDesc(const iir::StencilMetaInformation& metadata,
                               CodeGenProperties const& codeGenProperties)
    : ASTCodeGenCXX(), metadata_(metadata), codeGenProperties_(codeGenProperties) {}

ASTStencilDesc::~ASTStencilDesc() {}

std::string ASTStencilDesc::getName(const std::shared_ptr<iir::Stmt>& stmt) const {
  return metadata_.getFieldNameFromAccessID(*stmt->getData<iir::VarDeclStmtData>().AccessID);
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

  std::string stencilName =
      codeGenProperties_.getStencilName(StencilContext::SC_Stencil, stencilID);
  ss_ << "m_" << stencilName + "->run();\n";
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
