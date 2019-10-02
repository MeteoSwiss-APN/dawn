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

#include "dawn/CodeGen/GridTools/ASTStencilDesc.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/GridTools/CodeGenUtils.h"
#include "dawn/CodeGen/StencilFunctionAsBCGenerator.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace gt {

ASTStencilDesc::ASTStencilDesc(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const CodeGenProperties& codeGenProperties,
    const std::unordered_map<int, std::string>& stencilIdToArguments)
    : ASTCodeGenCXX(), instantiation_(stencilInstantiation),
      metadata_(stencilInstantiation->getMetaData()), codeGenProperties_(codeGenProperties),
      stencilIdToArguments_(stencilIdToArguments) {}

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

void ASTStencilDesc::visit(const std::shared_ptr<iir::BlockStmt>& stmt) { Base::visit(stmt); }

void ASTStencilDesc::visit(const std::shared_ptr<iir::ExprStmt>& stmt) { Base::visit(stmt); }

void ASTStencilDesc::visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
  dawn_unreachable("ReturnStmt not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) { Base::visit(stmt); }

void ASTStencilDesc::visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {
  dawn_unreachable("VerticalRegionDeclStmt not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {
  int stencilID = metadata_.getStencilIDFromStencilCallStmt(stmt);

  const auto& stencil = instantiation_->getIIR()->getStencil(stencilID);
  const auto fields = stencil.getOrderedFields();
  const auto& globalsMap = instantiation_->getIIR()->getGlobalVariableMap();
  auto plchdrs = CodeGenUtils::buildPlaceholderList(metadata_, fields, globalsMap, true);

  std::string stencilName =
      codeGenProperties_.getStencilName(StencilContext::SC_Stencil, stencilID);
  ss_ << std::string(indent_, ' ') << "m_" << stencilName
      << ".get_stencil()->run(" + RangeToString(",", "", "")(plchdrs) + "); " << std::endl;
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
  BCGenerator bcGen(metadata_, ss_);
  bcGen.generate(stmt);
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::IfStmt>& stmt) { Base::visit(stmt); }

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

// TODO use the forwarding visitor?
void ASTStencilDesc::visit(const std::shared_ptr<iir::UnaryOperator>& expr) { Base::visit(expr); }
void ASTStencilDesc::visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) {
  Base::visit(expr);
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::BinaryOperator>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<iir::AssignmentExpr>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<iir::TernaryOperator>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<iir::FunCallExpr>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
  dawn_unreachable("StencilFunCallExpr not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {
  dawn_unreachable("StencilFunArgExpr not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {
  if(metadata_.isAccessType(iir::FieldAccessType::FAT_GlobalVariable,
                            iir::getAccessIDFromExpr(expr)))
    ss_ << "m_globals.";

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) {
  Base::visit(expr);
}

void ASTStencilDesc::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  dawn_unreachable("FieldAccessExpr not allowed in StencilDesc AST");
}

} // namespace gt
} // namespace codegen
} // namespace dawn
