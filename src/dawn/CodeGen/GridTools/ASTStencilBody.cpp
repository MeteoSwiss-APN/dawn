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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilFunctionInstantiation.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace gt {

ASTStencilBody::ASTStencilBody(const StencilInstantiation* stencilInstantiation,
                               const std::unordered_map<Interval, std::string>& intervalToNameMap)
    : ASTCodeGenCXX(), instantiation_(stencilInstantiation), intervalToNameMap_(intervalToNameMap),
      offsetPrinter_(",", "(", ")"), currentFunction_(nullptr), nestingOfStencilFunArgLists_(0) {}

ASTStencilBody::~ASTStencilBody() {}

std::string ASTStencilBody::getName(const std::shared_ptr<Stmt>& stmt) const {
  if(currentFunction_)
    return currentFunction_->getNameFromAccessID(currentFunction_->getAccessIDFromStmt(stmt));
  else
    return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

std::string ASTStencilBody::getName(const std::shared_ptr<Expr>& expr) const {
  if(currentFunction_)
    return currentFunction_->getNameFromAccessID(currentFunction_->getAccessIDFromExpr(expr));
  else
    return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

int ASTStencilBody::getAccessID(const std::shared_ptr<Expr>& expr) const {
  if(currentFunction_)
    return currentFunction_->getAccessIDFromExpr(expr);
  else
    return instantiation_->getAccessIDFromExpr(expr);
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<BlockStmt>& stmt) { Base::visit(stmt); }

void ASTStencilBody::visit(const std::shared_ptr<ExprStmt>& stmt) { Base::visit(stmt); }

void ASTStencilBody::visit(const std::shared_ptr<ReturnStmt>& stmt) {
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

void ASTStencilBody::visit(const std::shared_ptr<VarDeclStmt>& stmt) { Base::visit(stmt); }

void ASTStencilBody::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  dawn_unreachable("VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  dawn_unreachable("StencilCallDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<IfStmt>& stmt) { Base::visit(stmt); }

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<UnaryOperator>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<BinaryOperator>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<AssignmentExpr>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<TernaryOperator>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<FunCallExpr>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  if(nestingOfStencilFunArgLists_++)
    ss_ << ", ";

  const StencilFunctionInstantiation* stencilFun =
      currentFunction_ ? currentFunction_->getStencilFunctionInstantiation(expr)
                       : instantiation_->getStencilFunctionInstantiation(expr);

  ss_ << "gridtools::call<" << StencilFunctionInstantiation::makeCodeGenName(*stencilFun) << ", "
      << intervalToNameMap_.find(stencilFun->getInterval())->second << ">::with(eval";

  for(auto& arg : expr->getArguments()) {
    arg->accept(*this);
  }

  // we record in a set all global variables passed to the stencil call
  std::set<int> globalVariablesInCallStmt;
  for(auto& arg : expr->getArguments()) {
    if(!isa<FieldAccessExpr>(*arg))
      continue;
    int accessID = currentFunction_ ? currentFunction_->getAccessIDFromExpr(arg)
                                    : instantiation_->getAccessIDFromExpr(arg);
    if(instantiation_->isGlobalVariable(accessID))
      globalVariablesInCallStmt.insert(accessID);
  }

  // explicitly add all global variables parameters that are used by stencil function but not passed
  // by user in stencil function call
  for(const int globalAccessID : stencilFun->getAccessIDSetGlobalVariables()) {
    if(globalVariablesInCallStmt.count(globalAccessID))
      continue;
    ss_ << "," << instantiation_->getNameFromAccessID(globalAccessID) << "()";
  }

  nestingOfStencilFunArgLists_--;
  ss_ << ")";
}

void ASTStencilBody::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {}

void ASTStencilBody::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int AccessID = getAccessID(expr);

  if(instantiation_->isGlobalVariable(AccessID)) {
    if(!nestingOfStencilFunArgLists_)
      ss_ << "eval(";
    else
      ss_ << ", ";

    ss_ << name << "()";

    if(!nestingOfStencilFunArgLists_)
      ss_ << ").value";

  } else {
    ss_ << name;

    if(expr->isArrayAccess()) {
      ss_ << "[";
      expr->getIndex()->accept(*this);
      ss_ << "]";
    }
  }
}

void ASTStencilBody::visit(const std::shared_ptr<LiteralAccessExpr>& expr) { Base::visit(expr); }

void ASTStencilBody::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  if(!nestingOfStencilFunArgLists_)
    ss_ << "eval(";
  else
    ss_ << ", ";

  if(currentFunction_) {
    ss_ << currentFunction_->getOriginalNameFromCallerAccessID(
               currentFunction_->getAccessIDFromExpr(expr))
        << offsetPrinter_(currentFunction_->evalOffsetOfFieldAccessExpr(expr, false));
  } else
    ss_ << getName(expr) << offsetPrinter_(expr->getOffset());

  if(!nestingOfStencilFunArgLists_)
    ss_ << ")";
}

void ASTStencilBody::setCurrentStencilFunction(
    const StencilFunctionInstantiation* currentFunction) {
  currentFunction_ = currentFunction;
}

} // namespace gt
} // namespace codegen
} // namespace dawn
