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

#include "dawn/CodeGen/CXXNaive-ico/ASTStencilBody.h"
#include "dawn/CodeGen/CXXNaive-ico/ASTStencilFunctionParamVisitor.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace cxxnaiveico {

ASTStencilBody::ASTStencilBody(const iir::StencilMetaInformation& metadata,
                               StencilContext stencilContext)
    : ASTCodeGenCXX(), metadata_(metadata), offsetPrinter_(",", "(", ")"),
      currentFunction_(nullptr), nestingOfStencilFunArgLists_(0), stencilContext_(stencilContext) {}

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

void ASTStencilBody::visit(const std::shared_ptr<iir::ExprStmt>& stmt) { Base::visit(stmt); }

void ASTStencilBody::visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}
void ASTStencilBody::visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) {
  auto getLocationTypeString = [](ast::Expr::LocationType type) {
    switch(type) {
    case ast::Expr::LocationType::Cells:
      return "Cell";
    case ast::Expr::LocationType::Edges:
      return "Edge";
    case ast::Expr::LocationType::Vertices:
      return "Node";
    default:
      dawn_unreachable("unknown location type");
      return "";
    }
  };
  std::string typeString = getLocationTypeString(expr->getRhsLocation());
  ss_ << std::string(indent_, ' ') << "reduce" + typeString + "ToCell(libtag_t(), m_mesh, t, ";
  expr->getInit()->accept(*this);
  ss_ << ", [&](auto& lhs, auto const& t) { return lhs " << expr->getOp() << "= ";
  auto argName = argName_;
  argName_ = "t";
  expr->getRhs()->accept(*this);
  argName_ = argName;
  ss_ << ";})";
}

void ASTStencilBody::visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) { Base::visit(stmt); }

void ASTStencilBody::visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
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

  const std::shared_ptr<iir::StencilFunctionInstantiation>& stencilFun =
      currentFunction_ ? currentFunction_->getStencilFunctionInstantiation(expr)
                       : metadata_.getStencilFunctionInstantiation(expr);

  ss_ << iir::StencilFunctionInstantiation::makeCodeGenName(*stencilFun) << "(i,j,k";

  ASTStencilFunctionParamVisitor fieldAccessVisitor(currentFunction_, metadata_);

  for(auto& arg : expr->getArguments()) {
    arg->accept(fieldAccessVisitor);
  }
  ss_ << fieldAccessVisitor.getCodeAndResetStream();

  nestingOfStencilFunArgLists_--;
  if(stencilFun->hasGlobalVariables()) {
    ss_ << ",m_globals";
  }
  ss_ << ")";
}

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {}

void ASTStencilBody::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int AccessID = getAccessID(expr);

  if(metadata_.isAccessType(iir::FieldAccessType::FAT_GlobalVariable, AccessID)) {
    ss_ << "m_globals." << name;
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

void ASTStencilBody::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {

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
      iir::StencilFunctionInstantiation& argStencilFn =
          *(currentFunction_->getFunctionInstantiationOfArgField(argIndex));

      ss_ << "m_"
          << iir::StencilFunctionInstantiation::makeCodeGenName(argStencilFn); // << "(i,j,k";

      /*
            // parse the arguments of the argument stencil gn call
            for(int argIdx = 0; argIdx < argStencilFn.numArgs(); ++argIdx) {
              // parse the argument if it is a field. Ignore offsets/directions,
              // since they are "inlined" in the code generation of the function
              if(argStencilFn.isArgField(argIdx)) {
                Array3i offset = currentFunction_->evalOffsetOfFieldAccessExpr(expr, false);

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
            */

    } else {
      std::string accessName = currentFunction_->getOriginalNameFromCallerAccessID(
          currentFunction_->getAccessIDFromExpr(expr));
      ss_ << "m_" + accessName;
      //<< offsetPrinter_(ijkfyOffset(currentFunction_->evalOffsetOfFieldAccessExpr(expr, false),
      // accessName));
    }
  } else {
    ss_ << "m_" << getName(expr) << "(" << argName_ << ")";
  }
}

void ASTStencilBody::setCurrentStencilFunction(
    const std::shared_ptr<iir::StencilFunctionInstantiation>& currentFunction) {
  currentFunction_ = currentFunction;
}

} // namespace cxxnaiveico
} // namespace codegen
} // namespace dawn
