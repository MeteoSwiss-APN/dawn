#include "UnstructuredDimensionChecker.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/AST/FieldDimension.h"
#include "dawn/AST/IterationSpace.h"
#include "dawn/AST/LocationType.h"
#include "dawn/AST/Offsets.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Stage.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Validator/WeightChecker.h"
#include <optional>

namespace dawn {

static const ast::UnstructuredFieldDimension& getUnstructuredDim(const ast::FieldDimensions& dims) {
  return ast::dimension_cast<const ast::UnstructuredFieldDimension&>(
      dims.getHorizontalFieldDimension());
}

static bool checkAgainstChain(const ast::UnstructuredFieldDimension& dim,
                              const ast::UnstructuredIterationSpace& space) {
  if(dim.isDense()) {
    return dim.getDenseLocationType() == space.Chain.back() ||
           dim.getDenseLocationType() == space.Chain.front();
  } else {
    return dim.getIterSpace() == space;
  }
}

UnstructuredDimensionChecker::ConsistencyResult
UnstructuredDimensionChecker::checkDimensionsConsistency(
    const dawn::iir::IIR& iir, const iir::StencilMetaInformation& metaData) {
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(iir)) {
    UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl checker(
        doMethod->getFieldDimensionsByName(), metaData.getAccessIDToNameMap(),
        metaData.getAccessIDToLocalVariableDataMap());
    for(const auto& stmt : doMethod->getAST().getStatements()) {
      stmt->accept(checker);
      if(!checker.isConsistent()) {
        return {false, stmt->getSourceLocation()};
      }
    }
  }
  return {true, SourceLocation()};
}

UnstructuredDimensionChecker::ConsistencyResult
UnstructuredDimensionChecker::checkDimensionsConsistency(const dawn::SIR& SIR) {
  // check type consistency of stencil functions
  for(auto const& stenFunIt : SIR.StencilFunctions) {
    std::unordered_map<std::string, ast::FieldDimensions> argumentFieldDimensions;
    for(const auto& arg : stenFunIt->Args) {
      if(arg->Kind == sir::StencilFunctionArg::ArgumentKind::Field) {
        const auto* argField = static_cast<sir::Field*>(arg.get());
        argumentFieldDimensions.insert({argField->Name, argField->Dimensions});
      }
    }
    for(const auto& ast : stenFunIt->Asts) {
      UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl checker(
          argumentFieldDimensions);
      for(const auto& stmt : ast->getRoot()->getChildren()) {
        stmt->accept(checker);
        if(!checker.isConsistent()) {
          return {false, stmt->getSourceLocation()};
        }
      }
    }
  }

  // check type consistency of stencils
  for(const auto& stencil : SIR.Stencils) {
    DAWN_ASSERT(stencil);
    std::unordered_map<std::string, ast::FieldDimensions> stencilFieldDims;
    for(const auto& field : stencil->Fields) {
      stencilFieldDims.insert({field->Name, field->Dimensions});
    }
    const auto& stencilAst = stencil->StencilDescAst;
    UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl checker(stencilFieldDims);
    for(const auto& stmt : stencilAst->getRoot()->getChildren()) {
      stmt->accept(checker);
      if(!checker.isConsistent()) {
        return {false, stmt->getSourceLocation()};
      }
    }
  }

  return {true, SourceLocation()};
}

UnstructuredDimensionChecker::ConsistencyResult
UnstructuredDimensionChecker::checkStageLocTypeConsistency(
    const iir::IIR& iir, const iir::StencilMetaInformation& metaData) {

  for(const auto& stage : iterateIIROver<iir::Stage>(iir)) {
    DAWN_ASSERT_MSG(stage->getLocationType().has_value(), "Location type of stage is unset.");
    auto stageLocationType = *stage->getLocationType();

    for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stage)) {
      for(const auto& stmt : doMethod->getAST().getStatements()) {
        UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl checker(
            doMethod->getFieldDimensionsByName(), metaData.getAccessIDToNameMap(),
            metaData.getAccessIDToLocalVariableDataMap());
        stmt->accept(checker);
        if(!(checker.hasHorizontalDimensions() &&
             stageLocationType ==
                 getUnstructuredDim(checker.getDimensions()).getDenseLocationType())) {
          return {false, stmt->getSourceLocation()};
        }
      }
    }
  }
  return {true, SourceLocation()};
}

UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::UnstructuredDimensionCheckerImpl(
    const std::unordered_map<std::string, ast::FieldDimensions> nameToDimensionsMap,
    UnstructuredDimensionCheckerConfig config)
    : nameToDimensions_(nameToDimensionsMap), config_(config) {
  checkType_ = checkType::runOnSIR;
}

UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::UnstructuredDimensionCheckerImpl(
    const std::unordered_map<std::string, ast::FieldDimensions> nameToDimensionsMap,
    const std::unordered_map<int, std::string> idToNameMap,
    const std::unordered_map<int, iir::LocalVariableData> idToLocalVariableData,
    UnstructuredDimensionCheckerConfig config)
    : nameToDimensions_(nameToDimensionsMap), idToNameMap_(idToNameMap),
      idToLocalVariableData_(idToLocalVariableData), config_(config) {
  checkType_ = checkType::runOnIIR;
}

const ast::FieldDimensions&
UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::getDimensions() const {
  DAWN_ASSERT(hasDimensions());
  return curDimensions_.value();
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::VarDeclStmt>& stmt) {
  if(!dimensionsConsistent_) {
    return;
  }
  if(checkType_ == checkType::runOnSIR) {
    return;
  }
  auto accessID = stmt->getData<iir::VarDeclStmtData>().AccessID;
  const auto varDeclInfo = idToLocalVariableData_.at(*accessID);
  // type is not set if PassLocalVarType didn't run
  if(!varDeclInfo.isTypeSet()) {
    return;
  }
  setCurDimensionFromLocType(varDeclInfo.getType());
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::FieldAccessExpr>& fieldAccessExpr) {
  if(!dimensionsConsistent_) {
    return;
  }

  auto fieldCompatible = [&](bool hasOffset, const ast::FieldDimensions& dims) {
    if(!dims.isVertical()) {
      if(hasOffset && getUnstructuredDim(dims).isDense()) {
        return getUnstructuredDim(dims).getDenseLocationType() ==
               config_.currentIterSpace_->Chain.back();
      }
      if(getUnstructuredDim(dims).isSparse()) {
        return getUnstructuredDim(dims).getIterSpace() == *config_.currentIterSpace_;
      }
    }
    return true;
  };

  bool hasOffset = ast::offset_cast<const ast::UnstructuredOffset&>(
                       fieldAccessExpr->getOffset().horizontalOffset())
                       .hasOffset();
  auto fieldName = fieldAccessExpr->getName();
  DAWN_ASSERT(nameToDimensions_.count(fieldName));
  curDimensions_ = nameToDimensions_.at(fieldName);
  dimensionsConsistent_ &= fieldCompatible(hasOffset, *curDimensions_);

  if(fieldAccessExpr->getOffset().hasVerticalIndirection()) {
    auto indirectionName = fieldAccessExpr->getOffset().getVerticalIndirectionFieldName();
    DAWN_ASSERT(nameToDimensions_.count(indirectionName));
    const auto& indirDims = nameToDimensions_.at(indirectionName);

    // if we aren't in an iteration (loop or reduction) the indirection needs to
    // match the field type...
    if(!config_.currentIterSpace_.has_value()) {
      dimensionsConsistent_ &= *curDimensions_ == indirDims;
    } else {
      //...otherwise we need to check against the chain
      checkAgainstChain(getUnstructuredDim(indirDims), *config_.currentIterSpace_);
    }
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::setCurDimensionFromLocType(
    iir::LocalVariableType&& type) {
  switch(type) {
  case iir::LocalVariableType::Scalar:
  case iir::LocalVariableType::OnIJ:
    return;
    break;
  case iir::LocalVariableType::OnCells:
    curDimensions_ = ast::FieldDimensions(
        ast::HorizontalFieldDimension{ast::unstructured, ast::LocationType::Cells}, true);
    break;
  case iir::LocalVariableType::OnEdges:
    curDimensions_ = ast::FieldDimensions(
        ast::HorizontalFieldDimension{ast::unstructured, ast::LocationType::Edges}, true);
    break;
  case iir::LocalVariableType::OnVertices:
    curDimensions_ = ast::FieldDimensions(
        ast::HorizontalFieldDimension{ast::unstructured, ast::LocationType::Vertices}, true);
    break;
  default:
    break;
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::VarAccessExpr>& varAccessExpr) {
  if(!dimensionsConsistent_) {
    return;
  }
  if(checkType_ == checkType::runOnSIR) {
    return;
  }
  DAWN_ASSERT(varAccessExpr->hasData());
  auto accessID = *varAccessExpr->getData<iir::IIRAccessExprData>().AccessID;
  // access may be global
  if(!idToLocalVariableData_.count(accessID)) {
    return;
  }
  const auto varAccessInfo = idToLocalVariableData_.at(accessID);
  if(!varAccessInfo.isTypeSet()) {
    // type is not set if PassLocalVarType didn't run
    return;
  }

  setCurDimensionFromLocType(varAccessInfo.getType());
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::IfStmt>& ifStmt) {
  visit(std::static_pointer_cast<ast::Stmt>(ifStmt));
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::BlockStmt>& blockStmt) {
  visit(std::static_pointer_cast<ast::Stmt>(blockStmt));
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::Stmt>& stmt) {
  std::optional<ast::FieldDimensions> prevDims = curDimensions_;
  for(auto& s : stmt->getChildren()) {
    s->accept(*this);
    if(isConsistent() && hasHorizontalDimensions()) {
      dimensionsConsistent_ &= (prevDims && !prevDims->isVertical())
                                   ? getUnstructuredDim(*prevDims).getDenseLocationType() ==
                                         getUnstructuredDim(*curDimensions_).getDenseLocationType()
                                   : true;
      prevDims = curDimensions_;
    }
  }
  if(!isConsistent())
    return;
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::checkBinaryOpUnstructured(
    const ast::FieldDimensions& left, const ast::FieldDimensions& right) {
  const auto& unstructuredDimLeft = getUnstructuredDim(left);
  const auto& unstructuredDimRight = getUnstructuredDim(right);

  if(config_.currentIterSpace_) {
    dimensionsConsistent_ &= checkAgainstChain(unstructuredDimLeft, *config_.currentIterSpace_);
    dimensionsConsistent_ &= checkAgainstChain(unstructuredDimRight, *config_.currentIterSpace_);
  } else {
    dimensionsConsistent_ &=
        unstructuredDimLeft.isDense() && unstructuredDimRight.isDense() &&
        unstructuredDimLeft.getDenseLocationType() == unstructuredDimRight.getDenseLocationType();
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::BinaryOperator>& binOp) {
  if(!dimensionsConsistent_) {
    return;
  }

  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl left(
      nameToDimensions_, idToNameMap_, idToLocalVariableData_, config_);
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl right(
      nameToDimensions_, idToNameMap_, idToLocalVariableData_, config_);

  binOp->getLeft()->accept(left);
  binOp->getRight()->accept(right);

  // type check failed further down below
  if(!(left.isConsistent() && right.isConsistent())) {
    dimensionsConsistent_ = false;
    return;
  }

  // if both sides access unstructured fields, they need to be on the same target location type
  // or both sides can be without type, e.g. 3*5 or 3*cell_field, so no error in this case
  if(left.hasHorizontalDimensions() && right.hasHorizontalDimensions()) {

    checkBinaryOpUnstructured(left.getDimensions(), right.getDimensions());

  } else if(left.hasHorizontalDimensions() && !right.hasHorizontalDimensions()) {
    curDimensions_ = left.getDimensions();
  } else if(!left.hasHorizontalDimensions() && right.hasHorizontalDimensions()) {
    curDimensions_ = right.getDimensions();
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::AssignmentExpr>& assignmentExpr) {
  if(!dimensionsConsistent_) {
    return;
  }
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl left(
      nameToDimensions_, idToNameMap_, idToLocalVariableData_, config_);
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl right(
      nameToDimensions_, idToNameMap_, idToLocalVariableData_, config_);

  assignmentExpr->getLeft()->accept(left);
  assignmentExpr->getRight()->accept(right);

  // type check failed further down below
  if(!(left.isConsistent() && right.isConsistent())) {
    dimensionsConsistent_ = false;
    return;
  }

  // assigning to sparse dimensions is only allowed in a foor loop context
  if(!config_.parentIsChainForLoop_ && left.hasHorizontalDimensions() &&
     getUnstructuredDim(left.getDimensions()).isSparse()) {
    dimensionsConsistent_ = false;
    return;
  }

  // assigning from sparse dimensions is only allowed in either reductions or for loops
  if(right.hasHorizontalDimensions() && getUnstructuredDim(right.getDimensions()).isSparse() &&
     !(config_.currentIterSpace_.has_value())) {
    dimensionsConsistent_ = false;
    return;
  }

  // if both sides access unstructured fields, they need to be on the same target location type
  // or both sides can be without type, e.g. 3*5 or 3*cell_field, so no error in this case
  if(left.hasHorizontalDimensions() && right.hasHorizontalDimensions()) {
    const auto& unstructuredDimLeft = getUnstructuredDim(left.getDimensions());
    const auto& unstructuredDimRight = getUnstructuredDim(right.getDimensions());

    // Case 0: Both lhs and rhs are sparse. Their dense + sparse parts must match.
    // example:
    // ```
    // field(edges, e->c->v) sparseL, sparseR;
    // ...
    // sparseL = sparseR;
    // ```
    // Result of assignment (= iteration space) is sparse dimension e->c->v
    if(unstructuredDimLeft.isSparse() && unstructuredDimRight.isSparse()) {
      // Check that neighbor chains match
      dimensionsConsistent_ =
          unstructuredDimLeft.getNeighborChain() == unstructuredDimRight.getNeighborChain();

    }
    // Lhs dense and rhs sparse is not possible, because there would be multiple values to write
    // on the same location.
    else if(unstructuredDimLeft.isDense() && unstructuredDimRight.isSparse()) {
      dimensionsConsistent_ = false;
    } else if(unstructuredDimLeft.isSparse() && unstructuredDimRight.isDense()) {

      // Case 1.a: Lhs is sparse, rhs is dense, rhs's location type matches lhs's target
      // location type (last of chain).
      // Example:
      // ```
      // field(edges, e->c->v) sparseL;
      // field(vertices) denseR;
      // ...
      // sparseL = denseR;
      // ```
      // Result of expression (= iteration space) is sparse dimension: e->c->v
      if(unstructuredDimLeft.getLastSparseLocationType() ==
         unstructuredDimRight.getDenseLocationType()) {
        // Nothing to do here
      }
      // Case 1.b: Lhs is sparse, rhs is dense, rhs's location type matches lhs's dense
      // location type (first of chain).
      // Example:
      // ```
      // field(edges, e->c->v) sparseL;
      // field(edges) denseR;
      // ...
      // sparseL = denseR;
      // ```
      // Result of expression (= iteration space) is sparse dimension: e->c->v
      else if(unstructuredDimLeft.getDenseLocationType() ==
              unstructuredDimRight.getDenseLocationType()) {
        // Nothing to do here

      } else { // No other subcase is allowed.
        dimensionsConsistent_ = false;
      }
    }
    // Case 2: Both operands are dense. They must match.
    // example:
    // ```
    // field(edges) denseL, denseR;
    // ...
    // denseL = denseR;
    // ```
    // Result of expression (= iteration space) is dense dimension edges
    else if(unstructuredDimLeft.isDense() && unstructuredDimRight.isDense()) {

      dimensionsConsistent_ =
          unstructuredDimLeft.getDenseLocationType() == unstructuredDimRight.getDenseLocationType();

    } else {
      dawn_unreachable("All cases should be covered.");
    }

    if(!dimensionsConsistent_) {
      return;
    }

    // Dimensions to propagate are always those of the lhs (if the lhs has dimensions)
    curDimensions_ = left.getDimensions();

  } else if(left.hasHorizontalDimensions() && !right.hasHorizontalDimensions()) {
    curDimensions_ = left.getDimensions();
  } else if(left.hasDimensions() && !left.hasHorizontalDimensions() &&
            right.hasHorizontalDimensions()) {
    dimensionsConsistent_ = false;
    return;
  } else if(!left.hasHorizontalDimensions() && right.hasHorizontalDimensions()) {
    // this may be ok, remember that PassLocalVar has not necessarily be run when this checker is
    // run
    curDimensions_ = right.getDimensions();
    return;
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::LoopStmt>& loopStmt) {
  config_.parentIsChainForLoop_ = true;
  const auto maybeChainPtr =
      dynamic_cast<const ast::ChainIterationDescr*>(loopStmt->getIterationDescrPtr());
  if(maybeChainPtr) {
    config_.currentIterSpace_ = maybeChainPtr->getIterSpace();
  }
  for(auto it : loopStmt->getChildren()) {
    it->accept(*this);
  }
  config_.parentIsChainForLoop_ = false;
  config_.currentIterSpace_ = std::nullopt;
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<ast::ReductionOverNeighborExpr>& reductionExpr) {
  if(!dimensionsConsistent_) {
    return;
  }

  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl init(
      nameToDimensions_, idToNameMap_, idToLocalVariableData_, config_);

  config_.currentIterSpace_ = reductionExpr->getIterSpace();
  reductionExpr->getInit()->accept(init);
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl ops(
      nameToDimensions_, idToNameMap_, idToLocalVariableData_, config_);
  reductionExpr->getRhs()->accept(ops);

  if(!ops.isConsistent()) {
    dimensionsConsistent_ = false;
    return;
  }

  // initial value needs to be consistent with operations on right hand side
  if(init.hasHorizontalDimensions() && ops.hasHorizontalDimensions()) {
    // As init and rhs get combined through a binary operation, let's reuse the same code
    checkBinaryOpUnstructured(init.getDimensions(), ops.getDimensions());
  }

  if(!dimensionsConsistent_) {
    return;
  }

  // check weighs for consistency w.r.t dimensions
  if(reductionExpr->getWeights().has_value()) {
    // check weights one by one
    UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl weightChecker(
        nameToDimensions_, idToNameMap_, idToLocalVariableData_, config_);
    for(const auto& weight : *reductionExpr->getWeights()) {
      weight->accept(weightChecker);
    }

    // if a weight is not consistent "in itself", abort
    if(!weightChecker.isConsistent()) {
      dimensionsConsistent_ = false;
      return;
    }

    // otherwise, all weights need to be of the same type, namely the lhs type of the reduction
    //  this assumes that all field accesses are dense in the weights. this restriction will
    //  eventually be lifted, but is for now ensured in the weights checker
    if(weightChecker.hasHorizontalDimensions()) {
      if(getUnstructuredDim(weightChecker.getDimensions()).getDenseLocationType() !=
         reductionExpr->getLhsLocation()) {
        dimensionsConsistent_ = false;
        return;
      }
    }
  }

  // if the rhs subtree has dimensions, we must check that such dimensions are consistent with the
  // declared rhs and lhs location types
  if(ops.hasHorizontalDimensions()) {
    const auto& rhsUnstructuredDim = getUnstructuredDim(ops.getDimensions());
    if(rhsUnstructuredDim.isSparse()) {
      dimensionsConsistent_ =
          (rhsUnstructuredDim.getLastSparseLocationType() == reductionExpr->getNbhChain().back() &&
           rhsUnstructuredDim.getDenseLocationType() == reductionExpr->getLhsLocation());
    } else {
      dimensionsConsistent_ =
          (rhsUnstructuredDim.getDenseLocationType() == reductionExpr->getNbhChain().back());
    }
  }

  if(!dimensionsConsistent_) {
    return;
  }

  // the reduce over neighbors concept imposes a type on the left hand side
  curDimensions_ = ast::FieldDimensions(
      ast::HorizontalFieldDimension(ast::unstructured, reductionExpr->getLhsLocation()),
      ops.hasHorizontalDimensions() ? ops.getDimensions().K() : false);
}

} // namespace dawn