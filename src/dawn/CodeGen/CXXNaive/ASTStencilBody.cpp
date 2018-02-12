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

#include "dawn/CodeGen/CXXNaive/ASTStencilBody.h"
#include "dawn/CodeGen/CXXNaive/ASTStencilFunctionParamVisitor.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilFunctionInstantiation.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace cxxnaive {

ASTStencilBody::ASTStencilBody(const dawn::StencilInstantiation* stencilInstantiation,
                               StencilContext stencilContext)
    : ASTCodeGenCXX(), instantiation_(stencilInstantiation), offsetPrinter_(",", "(", ")"),
      currentFunction_(nullptr), nestingOfStencilFunArgLists_(0), stencilContext_(stencilContext) {}

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

  ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTStencilBody::visit(const std::shared_ptr<VarDeclStmt>& stmt) { Base::visit(stmt); }

void ASTStencilBody::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
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

  const std::shared_ptr<dawn::StencilFunctionInstantiation>& stencilFun =
      currentFunction_ ? currentFunction_->getStencilFunctionInstantiation(expr)
                       : instantiation_->getStencilFunctionInstantiation(expr);

  ss_ << dawn::StencilFunctionInstantiation::makeCodeGenName(*stencilFun) << "(i,j,k";

  //  int n = 0;
  ASTStencilFunctionParamVisitor fieldAccessVisitor(currentFunction_, instantiation_);

  for(auto& arg : expr->getArguments()) {

    arg->accept(fieldAccessVisitor);
    //    ++n;
  }
  ss_ << fieldAccessVisitor.getCodeAndResetStream();

  nestingOfStencilFunArgLists_--;
  ss_ << ")";
}

void ASTStencilBody::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {}

void ASTStencilBody::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int AccessID = getAccessID(expr);

  if(instantiation_->isGlobalVariable(AccessID)) {
    ss_ << "globals::get()." << name << ".get_value()";
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

  if(currentFunction_) {
    // extract the arg index, from the AccessID
    int argIndex = -1;
    for(auto idx : currentFunction_->ArgumentIndexToCallerAccessIDMap()) {
      if(idx.second == currentFunction_->getAccessIDFromExpr(expr))
        argIndex = idx.first;
    }

    DAWN_ASSERT(argIndex != -1);

    // In order to explain the algorithm, let assume the following example
    // stencil_function fn {
    //   offset off1;
    //   storage st1;
    //   Do {
    //     st1(off1+2);
    //   }
    // }
    // ... Inside a stencil computation, there is a call to fn
    // fn(offset_fn1, gn(offset_gn1, storage2, storage3)) ;

    // If the arg index corresponds to a function instantation,
    // it means the function was called passing another function as argument,
    // i.e. gn in the example
    if(currentFunction_->isArgStencilFunctionInstantiation(argIndex)) {
      StencilFunctionInstantiation& argStencilFn =
          *(currentFunction_->getFunctionInstantiationOfArgField(argIndex));

      ss_ << StencilFunctionInstantiation::makeCodeGenName(argStencilFn) << "(i,j,k";

      // parse the arguments of the argument stencil gn call
      for(int argIdx = 0; argIdx < argStencilFn.numArgs(); ++argIdx) {
        // parse the argument if it is a field. Ignore offsets/directions,
        // since they are "inlined" in the code generation of the function
        if(argStencilFn.isArgField(argIdx)) {
          Array3i offset = expr->getOffset();

          // parse the offsets of the field access, that should be used to offset the evaluation
          // of gn(), i.e. off1+2 in the example
          for(auto idx : expr->getArgumentMap()) {
            if(idx != -1) {
              DAWN_ASSERT_MSG((argStencilFn.isArgOffset(idx)),
                              "index identified by argument map is not an offset arg");
              int dim = argStencilFn.getCallerOffsetOfArgOffset(idx)[0];
              int off = argStencilFn.getCallerOffsetOfArgOffset(idx)[1];
              DAWN_ASSERT(dim < offset.size());
              offset[dim] = off;
            }
          }

          std::string accessName =
              currentFunction_->getArgNameFromFunctionCall(argStencilFn.getName());
          ss_ << ", "
              << "pw_" + accessName << ".cloneWithOffset(std::array<int,"
              << std::to_string(offset.size()) << ">{";

          bool init = false;
          for(auto idxIt : offset) {
            if(init)
              ss_ << ",";
            ss_ << std::to_string(idxIt);
            init = true;
          }
          ss_ << "})";
        }
      }
      ss_ << ")";

    } else {
      std::string accessName = currentFunction_->getOriginalNameFromCallerAccessID(
          currentFunction_->getAccessIDFromExpr(expr));
      ss_ << accessName
          << offsetPrinter_(ijkfyOffset(currentFunction_->evalOffsetOfFieldAccessExpr(expr, false),
                                        accessName));
    }
  } else {
    std::string accessName = getName(expr);
    ss_ << accessName << offsetPrinter_(ijkfyOffset(expr->getOffset(), accessName));
  }
}

void ASTStencilBody::setCurrentStencilFunction(
    const std::shared_ptr<StencilFunctionInstantiation>& currentFunction) {
  currentFunction_ = currentFunction;
}

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
