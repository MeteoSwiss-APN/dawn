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
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace cxxnaiveico {

ASTStencilBody::ASTStencilBody(const iir::StencilMetaInformation& metadata,
                               StencilContext stencilContext)
    : ASTCodeGenCXX(), metadata_(metadata), offsetPrinter_(",", "(", ")"),
      currentFunction_(nullptr), nestingOfStencilFunArgLists_(0), stencilContext_(stencilContext) {}

ASTStencilBody::~ASTStencilBody() {}

std::string ASTStencilBody::getName(const std::shared_ptr<iir::VarDeclStmt>& stmt) const {
  if(currentFunction_)
    return currentFunction_->getFieldNameFromAccessID(iir::getAccessID(stmt));
  else
    return metadata_.getFieldNameFromAccessID(iir::getAccessID(stmt));
}

std::string ASTStencilBody::getName(const std::shared_ptr<iir::Expr>& expr) const {
  if(currentFunction_)
    return currentFunction_->getFieldNameFromAccessID(iir::getAccessID(expr));
  else
    return metadata_.getFieldNameFromAccessID(iir::getAccessID(expr));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}
void ASTStencilBody::visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) {
  auto getLocationTypeString = [](ast::LocationType type) {
    switch(type) {
    case ast::LocationType::Cells:
      return "Cell";
    case ast::LocationType::Edges:
      return "Edge";
    case ast::LocationType::Vertices:
      return "Vertex";
    default:
      dawn_unreachable("unknown location type");
      return "";
    }
  };

  std::string typeStringRHS = getLocationTypeString(expr->getRhsLocation());
  std::string typeStringLHS = getLocationTypeString(expr->getLhsLocation());

  bool hasWeights = expr->getWeights().has_value();

  std::string sigArg =
      (parentIsReduction_)
          ? "red_loc"
          : "loc"; // does stage or parent reduceOverNeighborExpr determine argname?
  ss_ << std::string(indent_, ' ')
      << "reduce" + typeStringRHS + "To" + typeStringLHS + "(LibTag{}, m_mesh," << sigArg << ", ";
  expr->getInit()->accept(*this);
  if(hasWeights) {
    ss_ << ", [&](auto& lhs, auto const& red_loc, auto const& weight) {\n";
    ss_ << "lhs " << expr->getOp() << "= ";
    ss_ << "weight * ";
  } else {
    ss_ << ", [&](auto& lhs, auto const& red_loc) { lhs " << expr->getOp() << "= ";
  }

  auto argName = denseArgName_;
  // arg names for dense and sparse location
  denseArgName_ = "red_loc";
  sparseArgName_ = "loc";
  // indicate if parent of subexpr is reduction
  parentIsReduction_ = true;
  expr->getRhs()->accept(*this);
  parentIsReduction_ = false;
  // "pop" argName
  denseArgName_ = argName;
  ss_ << ";\n";
  ss_ << "m_sparse_dimension_idx++;\n";
  ss_ << "return lhs;\n";
  ss_ << "}";
  if(hasWeights) {
    auto weights = expr->getWeights().value();
    bool first = true;
    auto typeStr = sir::Value::typeToString(weights[0].getType());
    ss_ << ", std::vector<" << typeStr << ">({";
    for(auto const& weight : weights) {
      if(!first) {
        ss_ << ", ";
      }
      DAWN_ASSERT_MSG(weight.has_value(), "weight with no value encountered in code generation!\n");
      ss_ << weight.toString();
      first = false;
    }

    ss_ << "})";
  }
  ss_ << ")";
}

void ASTStencilBody::visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

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
  int AccessID = iir::getAccessID(expr);

  if(metadata_.isAccessType(iir::FieldAccessType::GlobalVariable, AccessID)) {
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

void ASTStencilBody::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {

  if(currentFunction_) {
    // extract the arg index, from the AccessID
    int argIndex = -1;
    for(auto idx : currentFunction_->ArgumentIndexToCallerAccessIDMap()) {
      if(idx.second == iir::getAccessID(expr))
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
      std::string accessName =
          currentFunction_->getOriginalNameFromCallerAccessID(iir::getAccessID(expr));
      ss_ << "m_" + accessName;
      //<< offsetPrinter_(ijkfyOffset(currentFunction_->evalOffsetOfFieldAccessExpr(expr, false),
      // accessName));
    }
  } else {
    if(sir::dimension_cast<const sir::UnstructuredFieldDimension&>(
           metadata_.getFieldDimensions(iir::getAccessID(expr)).getHorizontalFieldDimension())
           .isDense()) {
      ss_ << "m_" << getName(expr) << "(deref(LibTag{}, " << denseArgName_ << "),"
          << "k+" << expr->getOffset().verticalOffset() << ")";
    } else {
      ss_ << "m_" << getName(expr) << "("
          << "deref(LibTag{}, " << sparseArgName_ << "),"
          << "m_sparse_dimension_idx, "
          << "k+" << expr->getOffset().verticalOffset() << ")";
    }
  }
}

void ASTStencilBody::setCurrentStencilFunction(
    const std::shared_ptr<iir::StencilFunctionInstantiation>& currentFunction) {
  currentFunction_ = currentFunction;
}

} // namespace cxxnaiveico
} // namespace codegen
} // namespace dawn
