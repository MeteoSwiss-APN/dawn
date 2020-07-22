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
#include "dawn/AST/Offsets.h"
#include "dawn/CodeGen/CXXNaive-ico/ASTStencilFunctionParamVisitor.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Unreachable.h"

static std::string nbhChainToVectorString(const std::vector<dawn::ast::LocationType>& chain) {
  auto getLocationTypeString = [](dawn::ast::LocationType type) {
    switch(type) {
    case dawn::ast::LocationType::Cells:
      return "dawn::LocationType::Cells";
    case dawn::ast::LocationType::Edges:
      return "dawn::LocationType::Edges";
    case dawn::ast::LocationType::Vertices:
      return "dawn::LocationType::Vertices";
    default:
      dawn_unreachable("unknown location type");
      return "";
    }
  };

  std::stringstream ss;
  ss << "std::vector<dawn::LocationType>{";
  bool first = true;
  for(const auto& loc : chain) {
    if(!first) {
      ss << ", ";
    }
    ss << getLocationTypeString(loc);
    first = false;
  }
  ss << "}";

  return ss.str();
}

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

void ASTStencilBody::visit(const std::shared_ptr<iir::BlockStmt>& stmt) {
  scopeDepth_++;
  ss_ << std::string(indent_, ' ') << "{\n";

  indent_ += DAWN_PRINT_INDENT;
  auto indent = std::string(indent_, ' ');
  for(const auto& s : stmt->getStatements()) {
    ss_ << indent;
    s->accept(*this);
  }
  indent_ -= DAWN_PRINT_INDENT;

  if(parentIsForLoop_) {
    ss_ << ASTStencilBody::LoopLinearIndexVarName() << "++;";
  }

  ss_ << std::string(indent_, ' ') << "}\n";
  scopeDepth_--;
}

void ASTStencilBody::visit(const std::shared_ptr<iir::LoopStmt>& stmt) {
  const auto maybeChainPtr =
      dynamic_cast<const ast::ChainIterationDescr*>(stmt->getIterationDescrPtr());
  DAWN_ASSERT_MSG(maybeChainPtr, "general loop concept not implemented yet!\n");

  ss_ << "{";
  ss_ << "int " << ASTStencilBody::LoopLinearIndexVarName() << " = 0;";
  ss_ << "for (auto " << ASTStencilBody::LoopNeighborIndexVarName()
      << ": getNeighbors(LibTag{}, m_mesh," << nbhChainToVectorString(maybeChainPtr->getChain())
      << ", " << ASTStencilBody::StageIndexVarName() << "))";
  parentIsForLoop_ = true;
  stmt->getBlockStmt()->accept(*this);
  parentIsForLoop_ = false;
  ss_ << "}";
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
    if(metadata_.getFieldDimensions(iir::getAccessID(expr)).isVertical()) {
      ss_ << "m_" << getName(expr) << "("
          << "k+" << expr->getOffset().verticalOffset() << ")";
      return;
    }

    bool isHorizontal = !metadata_.getFieldDimensions(iir::getAccessID(expr)).K();

    if(sir::dimension_cast<const sir::UnstructuredFieldDimension&>(
           metadata_.getFieldDimensions(iir::getAccessID(expr)).getHorizontalFieldDimension())
           .isDense()) {
      std::string resArgName = denseArgName_;
      if(!parentIsReduction_ && parentIsForLoop_) {
        resArgName =
            ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
                    .hasOffset()
                ? ASTStencilBody::LoopNeighborIndexVarName()
                : ASTStencilBody::StageIndexVarName();
      }
      ss_ << "m_" << getName(expr) << "(deref(LibTag{}, " << resArgName << ")";
      if(isHorizontal) {
        ss_ << ")";
      } else {
        ss_ << ",k+" << expr->getOffset().verticalOffset() << ")";
      }
    } else {
      std::string sparseIdx = parentIsReduction_
                                  ? ASTStencilBody::ReductionSparseIndexVarName(reductionDepth_ - 1)
                                  : ASTStencilBody::LoopLinearIndexVarName();
      ss_ << "m_" << getName(expr) << "("
          << "deref(LibTag{}, " << sparseArgName_ << ")," << sparseIdx;
      if(isHorizontal) {
        ss_ << ")";
      } else {
        ss_ << ",k+" << expr->getOffset().verticalOffset() << ")";
      }
    }
  }
}

void ASTStencilBody::visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) {
  bool hasWeights = expr->getWeights().has_value();

  std::string sigArg;
  if(parentIsReduction_) { // does stage or parent reduceOverNeighborExpr determine argname?
    sigArg = ASTStencilBody::ReductionIndexVarName(reductionDepth_);
  } else {
    if(parentIsForLoop_) {
      sigArg = ASTStencilBody::LoopNeighborIndexVarName();
    } else {
      sigArg = ASTStencilBody::StageIndexVarName();
    }
  }

  ss_ << std::string(indent_, ' ') << "reduce(LibTag{}, m_mesh," << sigArg << ", ";
  expr->getInit()->accept(*this);

  ss_ << ", " << nbhChainToVectorString(expr->getNbhChain());
  if(hasWeights) {
    ss_ << ", [&](auto& lhs, auto " << ASTStencilBody::ReductionIndexVarName(reductionDepth_ + 1)
        << ", auto const& weight) {\n";
    ss_ << "lhs " << expr->getOp() << "= ";
    ss_ << "weight * ";
  } else {
    ss_ << ", [&](auto& lhs, auto red_loc" << (reductionDepth_ + 1) << ") { ";
    // generate this next red_loc only if the rhs contains further reductions
    FindReduceOverNeighborExpr redFinder;
    expr->getRhs()->accept(redFinder);
    if(redFinder.hasReduceOverNeighborExpr()) {
      ss_ << "int " << ASTStencilBody::ReductionSparseIndexVarName(reductionDepth_ + 1) << " = 0;";
    }
    ss_ << "lhs " << expr->getOp() << "= ";
  }

  auto argName = denseArgName_;
  // arg names for dense and sparse location
  denseArgName_ =
      ASTStencilBody::ReductionIndexVarName(reductionDepth_ + 1); //<- always top of stack
  sparseArgName_ =
      (parentIsReduction_)
          ? ASTStencilBody::ReductionIndexVarName(reductionDepth_)
          : ASTStencilBody::StageIndexVarName(); //<- distincion: upper most level or not
  // indicate if parent of subexpr is reduction
  parentIsReduction_ = true;
  reductionDepth_++;
  expr->getRhs()->accept(*this);
  reductionDepth_--;
  parentIsReduction_ = false;
  // "pop" argName
  denseArgName_ = argName;
  ss_ << ";\n";
  ss_ << ASTStencilBody::ReductionSparseIndexVarName(reductionDepth_) << "++;\n";
  ss_ << "return lhs;\n";
  ss_ << "}";
  if(hasWeights) {
    auto weights = expr->getWeights().value();
    bool first = true;

    ss_ << ", std::vector<::dawn::float_type>({";
    for(auto const& weight : weights) {
      if(!first) {
        ss_ << ", ";
      }
      weight->accept(*this);
      first = false;
    }

    ss_ << "})";
  }
  ss_ << ")";
}

void ASTStencilBody::setCurrentStencilFunction(
    const std::shared_ptr<iir::StencilFunctionInstantiation>& currentFunction) {
  currentFunction_ = currentFunction;
}

} // namespace cxxnaiveico
} // namespace codegen
} // namespace dawn
