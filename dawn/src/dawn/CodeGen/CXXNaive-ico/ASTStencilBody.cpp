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
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Unreachable.h"

static std::string nbhChainToVectorString(const std::vector<dawn::ast::LocationType>& chain) {
  auto getLocationTypeString = [](dawn::ast::LocationType type) {
    switch(type) {
    case dawn::ast::LocationType::Cells:
      return "::dawn::LocationType::Cells";
    case dawn::ast::LocationType::Edges:
      return "::dawn::LocationType::Edges";
    case dawn::ast::LocationType::Vertices:
      return "::dawn::LocationType::Vertices";
    default:
      dawn_unreachable("unknown location type");
      return "";
    }
  };

  std::stringstream ss;
  ss << "std::vector<::dawn::LocationType>{";
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

std::string ASTStencilBody::getName(const std::shared_ptr<ast::VarDeclStmt>& stmt) const {
  if(currentFunction_)
    return currentFunction_->getFieldNameFromAccessID(iir::getAccessID(stmt));
  else
    return metadata_.getFieldNameFromAccessID(iir::getAccessID(stmt));
}

std::string ASTStencilBody::getName(const std::shared_ptr<ast::Expr>& expr) const {
  if(currentFunction_)
    return currentFunction_->getFieldNameFromAccessID(iir::getAccessID(expr));
  else
    return metadata_.getFieldNameFromAccessID(iir::getAccessID(expr));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<ast::ReturnStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::BlockStmt>& stmt) {
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

void ASTStencilBody::visit(const std::shared_ptr<ast::LoopStmt>& stmt) {
  const auto maybeChainPtr =
      dynamic_cast<const ast::ChainIterationDescr*>(stmt->getIterationDescrPtr());
  DAWN_ASSERT_MSG(maybeChainPtr, "general loop concept not implemented yet!\n");

  ss_ << "{";
  ss_ << "int " << ASTStencilBody::LoopLinearIndexVarName() << " = 0;";
  ss_ << "for (auto " << ASTStencilBody::LoopNeighborIndexVarName()
      << ": getNeighbors(LibTag{}, m_mesh," << nbhChainToVectorString(maybeChainPtr->getChain())
      << ", " << ASTStencilBody::StageIndexVarName()
      << (maybeChainPtr->getIncludeCenter() ? ",/*include center*/ true" : "") << "))";
  parentIsForLoop_ = true;
  currentChain_ = maybeChainPtr->getChain();
  stmt->getBlockStmt()->accept(*this);
  currentChain_.clear();
  parentIsForLoop_ = false;
  ss_ << "}";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<ast::StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<ast::BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<ast::StencilFunCallExpr>& expr) {
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

void ASTStencilBody::visit(const std::shared_ptr<ast::StencilFunArgExpr>& expr) {}

void ASTStencilBody::visit(const std::shared_ptr<ast::VarAccessExpr>& expr) {
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

std::string ASTStencilBody::makeIndexString(const std::shared_ptr<ast::FieldAccessExpr>& expr,
                                            std::string kiterStr) {
  bool isVertical = metadata_.getFieldDimensions(iir::getAccessID(expr)).isVertical();
  if(isVertical) {
    return kiterStr;
  }

  bool isHorizontal = !metadata_.getFieldDimensions(iir::getAccessID(expr)).K();
  bool isFullField = !isHorizontal && !isVertical;
  auto unstrDims = ast::dimension_cast<const ast::UnstructuredFieldDimension&>(
      metadata_.getFieldDimensions(iir::getAccessID(expr)).getHorizontalFieldDimension());
  bool isDense = unstrDims.isDense();
  bool isSparse = unstrDims.isSparse();

  if(isFullField && isDense) {
    std::string resArgName = denseArgName_;
    if((parentIsForLoop_ || parentIsReduction_) && reductionDepth_ == 1) {
      if(unstrDims.getDenseLocationType() == currentChain_.front() &&
         !ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
              .hasOffset()) {
        resArgName = "loc";
      }
    }
    if(!parentIsReduction_ && parentIsForLoop_) {
      resArgName =
          ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
                  .hasOffset()
              ? ASTStencilBody::LoopNeighborIndexVarName()
              : ASTStencilBody::StageIndexVarName();
    }
    return "deref(LibTag{}, " + resArgName + "), " + kiterStr;
  }

  if(isFullField && isSparse) {
    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");
    std::string sparseIdx = parentIsReduction_
                                ? ASTStencilBody::ReductionSparseIndexVarName(reductionDepth_ - 1)
                                : ASTStencilBody::LoopLinearIndexVarName();
    return "deref(LibTag{}, " + sparseArgName_ + ")," + sparseIdx + ", " + kiterStr;
  }

  if(isHorizontal && isDense) {
    std::string resArgName = denseArgName_;
    if((parentIsForLoop_ || parentIsReduction_) && reductionDepth_ == 1) {
      if(unstrDims.getDenseLocationType() == currentChain_.front() &&
         !ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
              .hasOffset()) {
        resArgName = "loc";
      }
    }
    if(!parentIsReduction_ && parentIsForLoop_) {
      resArgName =
          ast::offset_cast<const ast::UnstructuredOffset&>(expr->getOffset().horizontalOffset())
                  .hasOffset()
              ? ASTStencilBody::LoopNeighborIndexVarName()
              : ASTStencilBody::StageIndexVarName();
    }
    return "deref(LibTag{}, " + resArgName + ")";
  }

  if(isHorizontal && isSparse) {
    DAWN_ASSERT_MSG(parentIsForLoop_ || parentIsReduction_,
                    "Sparse Field Access not allowed in this context");
    std::string sparseIdx = parentIsReduction_
                                ? ASTStencilBody::ReductionSparseIndexVarName(reductionDepth_ - 1)
                                : ASTStencilBody::LoopLinearIndexVarName();
    return "deref(LibTag{}, " + sparseArgName_ + ")," + sparseIdx;
  }

  DAWN_ASSERT_MSG(false, "Bad Field configuration found in code gen!");
  return "BAD_FIELD_CONFIG";
}

void ASTStencilBody::visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) {

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
    if(!expr->getOffset().hasVerticalIndirection()) {
      ss_ << "m_"
          << expr->getName() + "(" +
                 makeIndexString(expr, "(k + " + std::to_string(expr->getOffset().verticalShift()) +
                                           ")") +
                 ")";
    } else {
      auto vertOffset = makeIndexString(std::static_pointer_cast<ast::FieldAccessExpr>(
                                            expr->getOffset().getVerticalIndirectionFieldAsExpr()),
                                        "k");
      ss_ << "m_"
          << expr->getName() + "(" +
                 makeIndexString(expr, "(m_" + expr->getOffset().getVerticalIndirectionFieldName() +
                                           "(" + vertOffset + ")" + +" + " +
                                           std::to_string(expr->getOffset().verticalShift()) +
                                           ")") +
                 ")";
    }
  }
}

void ASTStencilBody::visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) {
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
    ss_ << ", [&, " + ASTStencilBody::ReductionSparseIndexVarName(reductionDepth_) +
               " = int(0)](auto& "
               "lhs, auto "
        << ASTStencilBody::ReductionIndexVarName(reductionDepth_ + 1)
        << ", auto const& weight) mutable {\n";
  } else {
    ss_ << ", [&, " + ASTStencilBody::ReductionSparseIndexVarName(reductionDepth_) +
               " = int(0)](auto& lhs, auto "
        << ASTStencilBody::ReductionIndexVarName(reductionDepth_ + 1) << ") mutable { ";
  }

  if(!expr->isArithmetic()) {
    ss_ << "lhs = " << expr->getOp() << "(lhs, ";
  } else {
    ss_ << "lhs " << expr->getOp() << "= ";
  }

  if(hasWeights) {
    ss_ << "weight * ";
  }

  auto argName = denseArgName_;
  // arg names for dense and sparse location
  // if(parentIsReduction_) {
  denseArgName_ =
      ASTStencilBody::ReductionIndexVarName(reductionDepth_ + 1); //<- always top of stack
  // }
  sparseArgName_ =
      (parentIsReduction_)
          ? ASTStencilBody::ReductionIndexVarName(reductionDepth_)
          : ASTStencilBody::StageIndexVarName(); //<- distincion: upper most level or not
  // indicate if parent of subexpr is reduction
  parentIsReduction_ = true;
  currentChain_ = expr->getNbhChain();
  reductionDepth_++;
  expr->getRhs()->accept(*this);
  reductionDepth_--;
  if(reductionDepth_ == 0) {
    parentIsReduction_ = false;
    currentChain_.clear();
  }
  // "pop" argName
  denseArgName_ = argName;
  if(!expr->isArithmetic()) {
    ss_ << ")";
  }
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
  if(expr->getIncludeCenter()) {
    ss_ << ", /*include center*/ true";
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
