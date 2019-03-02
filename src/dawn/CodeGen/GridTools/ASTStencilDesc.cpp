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
#include "dawn/CodeGen/StencilFunctionAsBCGenerator.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace gt {

ASTStencilDesc::ASTStencilDesc(const std::shared_ptr<iir::StencilInstantiation> instantiation,
                               const CodeGenProperties& codeGenProperties,
                               const std::unordered_map<int, std::string>& stencilIdToArguments)
    : ASTCodeGenCXX(), instantiation_(instantiation), codeGenProperties_(codeGenProperties),
      stencilIdToArguments_(stencilIdToArguments) {}

ASTStencilDesc::~ASTStencilDesc() {}

std::string ASTStencilDesc::getName(const std::shared_ptr<Stmt>& stmt) const {
  return instantiation_->getFieldNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

std::string ASTStencilDesc::getName(const std::shared_ptr<Expr>& expr) const {
  return instantiation_->getFieldNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilDesc::visit(const std::shared_ptr<BlockStmt>& stmt) { Base::visit(stmt); }

void ASTStencilDesc::visit(const std::shared_ptr<ExprStmt>& stmt) { Base::visit(stmt); }

void ASTStencilDesc::visit(const std::shared_ptr<ReturnStmt>& stmt) {
  dawn_unreachable("ReturnStmt not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<VarDeclStmt>& stmt) { Base::visit(stmt); }

void ASTStencilDesc::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  dawn_unreachable("VerticalRegionDeclStmt not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  int StencilID = instantiation_->getStencilIDFromStmt(stmt);

  std::string stencilName =
      codeGenProperties_.getStencilName(StencilContext::SC_Stencil, StencilID);
  ss_ << std::string(indent_, ' ') << "m_" << stencilName << ".get_stencil()->run();\n";
}

void ASTStencilDesc::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
  BCGenerator bcGen(instantiation_, ss_);
  bcGen.generate(stmt);
}

void ASTStencilDesc::visit(const std::shared_ptr<IfStmt>& stmt) { Base::visit(stmt); }

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilDesc::visit(const std::shared_ptr<UnaryOperator>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<BinaryOperator>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<AssignmentExpr>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<TernaryOperator>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<FunCallExpr>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  dawn_unreachable("StencilFunCallExpr not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
  dawn_unreachable("StencilFunArgExpr not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  if(instantiation_->isGlobalVariable(instantiation_->getAccessIDFromExpr(expr)))
    ss_ << "m_globals.";

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

void ASTStencilDesc::visit(const std::shared_ptr<LiteralAccessExpr>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  dawn_unreachable("FieldAccessExpr not allowed in StencilDesc AST");
}

} // namespace gt
} // namespace codegen
} // namespace dawn
