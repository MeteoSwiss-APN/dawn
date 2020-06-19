#include "UnstructuredDimensionChecker.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/AST/Offsets.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Stage.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Validator/WeightChecker.h"

namespace dawn {

static const sir::UnstructuredFieldDimension& getUnstructuredDim(const sir::FieldDimensions& dims) {
  return sir::dimension_cast<const sir::UnstructuredFieldDimension&>(
      dims.getHorizontalFieldDimension());
}

UnstructuredDimensionChecker::ConsistencyResult
UnstructuredDimensionChecker::checkDimensionsConsistency(
    const dawn::iir::IIR& iir, const iir::StencilMetaInformation& metaData) {
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(iir)) {
    UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl checker(
        doMethod->getFieldDimensionsByName(), metaData.getAccessIDToNameMap());
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
    std::unordered_map<std::string, sir::FieldDimensions> argumentFieldDimensions;
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
    std::unordered_map<std::string, sir::FieldDimensions> stencilFieldDims;
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
            doMethod->getFieldDimensionsByName(), metaData.getAccessIDToNameMap());
        stmt->accept(checker);
        if(!(checker.hasDimensions() &&
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
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap,
    UnstructuredDimensionCheckerConfig config)
    : nameToDimensions_(nameToDimensionsMap), config_(config) {}

UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::UnstructuredDimensionCheckerImpl(
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap,
    const std::unordered_map<int, std::string> idToNameMap,
    UnstructuredDimensionCheckerConfig config)
    : nameToDimensions_(nameToDimensionsMap), idToNameMap_(idToNameMap), config_(config) {}

const sir::FieldDimensions&
UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::getDimensions() const {
  DAWN_ASSERT(hasDimensions());
  return curDimensions_.value();
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<iir::FieldAccessExpr>& fieldAccessExpr) {
  if(!dimensionsConsistent_) {
    return;
  }

  bool hasOffset = ast::offset_cast<const ast::UnstructuredOffset&>(
                       fieldAccessExpr->getOffset().horizontalOffset())
                       .hasOffset();

  auto fieldName = fieldAccessExpr->getName();
  // the name in the FieldAccessExpr may be stale if the there are nested stencils
  // in this case we need to look up the new AccessID in the data of the fieldAccessExpr
  if(fieldAccessExpr->hasData()) {
    auto newAccessID = fieldAccessExpr->getData<iir::IIRAccessExprData>().AccessID;
    if(newAccessID.has_value()) {
      DAWN_ASSERT(idToNameMap_.count(newAccessID.value()));
      fieldName = idToNameMap_.at(newAccessID.value());
    }
  }

  DAWN_ASSERT(nameToDimensions_.count(fieldName));
  curDimensions_ = nameToDimensions_.at(fieldName);

  if(hasOffset && getUnstructuredDim(*curDimensions_).isDense()) {
    dimensionsConsistent_ &=
        getUnstructuredDim(*curDimensions_).getDenseLocationType() == config_.currentChain_->back();
  }
}

static bool checkAgainstChain(const sir::UnstructuredFieldDimension& dim,
                              ast::NeighborChain chain) {
  if(dim.isDense()) {
    return dim.getDenseLocationType() == chain.back() ||
           dim.getDenseLocationType() == chain.front();
  } else {
    return dim.getNeighborChain() == chain;
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::checkBinaryOpUnstructured(
    const sir::FieldDimensions& left, const sir::FieldDimensions& right) {
  const auto& unstructuredDimLeft = getUnstructuredDim(left);
  const auto& unstructuredDimRight = getUnstructuredDim(right);

  if(config_.currentChain_) {
    dimensionsConsistent_ &= checkAgainstChain(unstructuredDimLeft, *config_.currentChain_);
    dimensionsConsistent_ &= checkAgainstChain(unstructuredDimRight, *config_.currentChain_);
  } else {
    dimensionsConsistent_ &=
        unstructuredDimLeft.isDense() && unstructuredDimRight.isDense() &&
        unstructuredDimLeft.getDenseLocationType() == unstructuredDimRight.getDenseLocationType();
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<iir::BinaryOperator>& binOp) {
  if(!dimensionsConsistent_) {
    return;
  }

  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl left(nameToDimensions_,
                                                                      idToNameMap_, config_);
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl right(nameToDimensions_,
                                                                       idToNameMap_, config_);

  binOp->getLeft()->accept(left);
  binOp->getRight()->accept(right);

  // type check failed further down below
  if(!(left.isConsistent() && right.isConsistent())) {
    dimensionsConsistent_ = false;
    return;
  }

  // if both sides access unstructured fields, they need to be on the same target location type
  // or both sides can be without type, e.g. 3*5 or 3*cell_field, so no error in this case
  if(left.hasDimensions() && right.hasDimensions()) {

    checkBinaryOpUnstructured(left.getDimensions(), right.getDimensions());

  } else if(left.hasDimensions() && !right.hasDimensions()) {
    curDimensions_ = left.getDimensions();
  } else if(!left.hasDimensions() && right.hasDimensions()) {
    curDimensions_ = right.getDimensions();
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<iir::AssignmentExpr>& assignmentExpr) {
  if(!dimensionsConsistent_) {
    return;
  }
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl left(nameToDimensions_,
                                                                      idToNameMap_, config_);
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl right(nameToDimensions_,
                                                                       idToNameMap_, config_);

  assignmentExpr->getLeft()->accept(left);
  assignmentExpr->getRight()->accept(right);

  // type check failed further down below
  if(!(left.isConsistent() && right.isConsistent())) {
    dimensionsConsistent_ = false;
    return;
  }

  // assigning to sparse dimensions is only allowed in a foor loop context
  if(!config_.parentIsChainForLoop_ && left.hasDimensions() &&
     getUnstructuredDim(left.getDimensions()).isSparse()) {
    dimensionsConsistent_ = false;
    return;
  }

  // assigning from sparse dimensions is only allowed in either reductions or for loops
  if(right.hasDimensions() && getUnstructuredDim(right.getDimensions()).isSparse() &&
     !(config_.parentIsReduction_ || config_.parentIsChainForLoop_)) {
    dimensionsConsistent_ = false;
    return;
  }

  // if both sides access unstructured fields, they need to be on the same target location type
  // or both sides can be without type, e.g. 3*5 or 3*cell_field, so no error in this case
  if(left.hasDimensions() && right.hasDimensions()) {

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

  } else if(left.hasDimensions() && !right.hasDimensions()) {
    curDimensions_ = left.getDimensions();
  } else if(!left.hasDimensions() && right.hasDimensions()) {
    curDimensions_ = right.getDimensions();
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<iir::LoopStmt>& loopStmt) {
  config_.parentIsChainForLoop_ = true;
  const auto maybeChainPtr =
      dynamic_cast<const ast::ChainIterationDescr*>(loopStmt->getIterationDescrPtr());
  if(maybeChainPtr) {
    config_.currentChain_ = maybeChainPtr->getChain();
  }
  for(auto it : loopStmt->getChildren()) {
    it->accept(*this);
  }
  config_.parentIsChainForLoop_ = false;
  config_.currentChain_ = std::nullopt;
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<iir::ReductionOverNeighborExpr>& reductionExpr) {
  if(!dimensionsConsistent_) {
    return;
  }

  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl init(nameToDimensions_,
                                                                      idToNameMap_, config_);

  config_.parentIsReduction_ = true;
  config_.currentChain_ = reductionExpr->getNbhChain();
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl ops(nameToDimensions_,
                                                                     idToNameMap_, config_);
  config_.currentChain_ = std::nullopt;
  reductionExpr->getInit()->accept(init);
  reductionExpr->getRhs()->accept(ops);

  if(!ops.isConsistent()) {
    dimensionsConsistent_ = false;
    return;
  }

  // initial value needs to be consistent with operations on right hand side
  if(init.hasDimensions() && ops.hasDimensions()) {
    // As init and rhs get combined through a binary operation, let's reuse the same code
    checkBinaryOpUnstructured(init.getDimensions(), ops.getDimensions());
  }

  if(!dimensionsConsistent_) {
    return;
  }

  // check weighs for consistency w.r.t dimensions
  if(reductionExpr->getWeights().has_value()) {
    // check weights one by one
    UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl weightChecker(nameToDimensions_,
                                                                                 idToNameMap_);
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
    if(weightChecker.hasDimensions()) {
      if(getUnstructuredDim(weightChecker.getDimensions()).getDenseLocationType() !=
         reductionExpr->getLhsLocation()) {
        dimensionsConsistent_ = false;
        return;
      }
    }
  }

  // if the rhs subtree has dimensions, we must check that such dimensions are consistent with the
  // declared rhs and lhs location types
  if(ops.hasDimensions()) {
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
  curDimensions_ = sir::FieldDimensions(
      sir::HorizontalFieldDimension(ast::unstructured, reductionExpr->getLhsLocation()),
      ops.hasDimensions() ? ops.getDimensions().K() : false);
}

} // namespace dawn