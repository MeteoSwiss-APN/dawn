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

#include "dawn/CodeGen/GridTools/ASTStencilBody.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace gt {

ASTStencilBody::ASTStencilBody(
    const iir::StencilMetaInformation& metadata,
    const std::unordered_set<iir::IntervalProperties>& intervalProperties)
    : ASTCodeGenCXX(), metadata_(metadata), intervalProperties_(intervalProperties),
      currentFunction_(nullptr), nestingOfStencilFunArgLists_(0) {}

ASTStencilBody::~ASTStencilBody() {}

std::string ASTStencilBody::getName(const std::shared_ptr<iir::Stmt>& stmt) const {
  if(currentFunction_)
    return currentFunction_->getFieldNameFromAccessID(currentFunction_->getAccessIDFromStmt(stmt));
  else
    return metadata_.getFieldNameFromAccessID(metadata_.getAccessIDFromStmt(stmt));
}

std::string ASTStencilBody::getName(const std::shared_ptr<iir::Expr>& expr) const {
  if(currentFunction_)
    return currentFunction_->getFieldNameFromAccessID(currentFunction_->getAccessIDFromExpr(expr));
  else
    return metadata_.getFieldNameFromAccessID(metadata_.getAccessIDFromExpr(expr));
}

int ASTStencilBody::getAccessID(const std::shared_ptr<iir::Expr>& expr) const {
  if(currentFunction_)
    return currentFunction_->getAccessIDFromExpr(expr);
  else
    return metadata_.getAccessIDFromExpr(expr);
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<iir::BlockStmt>& stmt) { Base::visit(stmt); }

void ASTStencilBody::visit(const std::shared_ptr<iir::ExprStmt>& stmt) {
  if(isa<iir::StencilFunCallExpr>(*(stmt->getExpr())))
    triggerCallProc_ = true;
  Base::visit(stmt);
}

void ASTStencilBody::visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  // Inside stencil functions there are no return statements, instead we assign the return value to
  // the special field `__out`
  if(currentFunction_)
    ss_ << "eval(__out(0, 0, 0)) =";
  else
    ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTStencilBody::visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) { Base::visit(stmt); }

void ASTStencilBody::visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {
  dawn_unreachable("VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {
  dawn_unreachable("StencilCallDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::IfStmt>& stmt) { Base::visit(stmt); }

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<iir::UnaryOperator>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<iir::BinaryOperator>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<iir::AssignmentExpr>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<iir::TernaryOperator>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<iir::FunCallExpr>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
  if(nestingOfStencilFunArgLists_++)
    ss_ << ", ";

  const std::shared_ptr<iir::StencilFunctionInstantiation> stencilFun =
      currentFunction_ ? currentFunction_->getStencilFunctionInstantiation(expr)
                       : metadata_.getStencilFunctionInstantiation(expr);

  ss_ << (triggerCallProc_ ? "gridtools::call_proc<" : "gridtools::call<")
      << iir::StencilFunctionInstantiation::makeCodeGenName(*stencilFun) << ", "
      << intervalProperties_.find(stencilFun->getInterval())->name_ << ">::with(eval";

  triggerCallProc_ = false;

  for(auto& arg : expr->getArguments()) {
    arg->accept(*this);
  }

  if(stencilFun->hasGlobalVariables()) {
    ss_ << ","
        << "globals()";
  }

  nestingOfStencilFunArgLists_--;
  ss_ << ")";
}

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {}

void ASTStencilBody::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int AccessID = getAccessID(expr);

  if(metadata_.isAccessType(iir::FieldAccessType::FAT_GlobalVariable, AccessID)) {
    if(!nestingOfStencilFunArgLists_)
      ss_ << "eval(";
    else
      ss_ << ", ";

    ss_ << "globals()";

    if(!nestingOfStencilFunArgLists_) {
      ss_ << ")." << name;
    }
  } else {
    ss_ << name;

    if(expr->isArrayAccess()) {
      ss_ << "[";
      expr->getIndex()->accept(*this);
      ss_ << "]";
    }
  }
}

void ASTStencilBody::visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) {
  Base::visit(expr);
}

static void printOffset(std::ostream& os, ast::Offset const& offset) {
  auto const& hoffset = ast::offset_cast<ast::StructuredOffset const&>(offset.horizontalOffset());
  auto const& voffset = offset.verticalOffset();

  os << "(" << hoffset.offsetI() << "," << hoffset.offsetJ() << "," << voffset << ")";
}

void ASTStencilBody::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  if(!nestingOfStencilFunArgLists_)
    ss_ << "eval(";
  else
    ss_ << ", ";

  if(currentFunction_) {
    ss_ << currentFunction_->getOriginalNameFromCallerAccessID(
        currentFunction_->getAccessIDFromExpr(expr));
    printOffset(ss_, currentFunction_->evalOffsetOfFieldAccessExpr(expr, false));
  } else {
    ss_ << getName(expr);
    printOffset(ss_, expr->getOffset());
  }

  if(!nestingOfStencilFunArgLists_)
    ss_ << ")";
}

void ASTStencilBody::setCurrentStencilFunction(
    const std::shared_ptr<iir::StencilFunctionInstantiation>& currentFunction) {
  currentFunction_ = currentFunction;
}

} // namespace gt
} // namespace codegen
} // namespace dawn
