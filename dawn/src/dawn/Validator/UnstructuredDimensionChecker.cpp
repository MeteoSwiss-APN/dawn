#include "UnstructuredDimensionChecker.h"

namespace dawn {

bool UnstructuredDimensionChecker::checkDimensionsConsistency(
    const dawn::iir::IIR& iir, const iir::StencilMetaInformation& metaData) {
  for(const auto& doMethodPtr : iterateIIROver<iir::DoMethod>(iir)) {
    UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl checker(
        doMethodPtr->getFieldDimensionsByName(), metaData.getAccessIDToNameMap());
    const auto& ast = doMethodPtr->getASTPtr();
    ast->accept(checker);
    if(!checker.isConsistent()) {
      return false;
    }
  }
  return true;
}

bool UnstructuredDimensionChecker::checkDimensionsConsistency(const dawn::SIR& SIR) {
  // check type consistency of stencil functions
  for(auto const& stenFunIt : SIR.StencilFunctions) {
    std::unordered_map<std::string, sir::FieldDimensions> argumentFieldDimensions;
    for(const auto& arg : stenFunIt->Args) {
      if(arg->Kind == sir::StencilFunctionArg::ArgumentKind::Field) {
        const auto* argField = static_cast<sir::Field*>(arg.get());
        argumentFieldDimensions.insert({argField->Name, argField->Dimensions});
      }
    }
    for(const auto& astIt : stenFunIt->Asts) {
      UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl checker(
          argumentFieldDimensions);
      astIt->accept(checker);
      if(!checker.isConsistent()) {
        return false;
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
    stencilAst->accept(checker);
    if(!checker.isConsistent()) {
      return false;
    }
  }

  return true;
}

///@brief Helper functions
namespace {

const sir::UnstructuredFieldDimension& getUnstructuredDim(const sir::FieldDimensions& dims) {
  return sir::dimension_cast<const sir::UnstructuredFieldDimension&>(
      dims.getHorizontalFieldDimension());
}

} // namespace

UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::UnstructuredDimensionCheckerImpl(
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap)
    : nameToDimensions_(nameToDimensionsMap) {}

UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::UnstructuredDimensionCheckerImpl(
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap,
    const std::unordered_map<int, std::string> idToNameMap)
    : nameToDimensions_(nameToDimensionsMap), idToNameMap_(idToNameMap) {}

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
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::checkBinaryOpUnstructured(
    const sir::FieldDimensions& left, const sir::FieldDimensions& right) {
  const auto& unstructuredDimLeft = getUnstructuredDim(left);
  const auto& unstructuredDimRight = getUnstructuredDim(right);

  // Case 0: Both operands are sparse. Their dense + sparse parts must match.
  // example:
  // ```
  // field(edges, e->c->v) sparseL, sparseR;
  // ...
  // expr = (sparseL + sparseR);
  // ```
  // Result of expression (= iteration space) is sparse dimension e->c->v
  if(unstructuredDimLeft.isSparse() && unstructuredDimRight.isSparse()) {
    // Check that neighbor chains match
    if(unstructuredDimLeft.getNeighborChain() != unstructuredDimRight.getNeighborChain()) {
      dimensionsConsistent_ = false;
      return;
    }
    curDimensions_ = left; // pick one, they are the same
  } else if(unstructuredDimLeft.isDense() && unstructuredDimRight.isSparse()) {

    // Case 1.a: Left is dense, right is sparse, left's location type matches right's target
    // location type (last of chain).
    // Example:
    // ```
    // field(vertices) denseL;
    // field(edges, e->c->v) sparseR;
    // ...
    // expr = (denseL + sparseR);
    // ```
    // Result of expression (= iteration space) is sparse dimension: e->c->v
    if(unstructuredDimLeft.getDenseLocationType() ==
       unstructuredDimRight.getLastSparseLocationType()) {
      // Propagate sparse
      curDimensions_ = right;
    }
    // Case 1.b: Left is dense, right is sparse, left's location type matches right's dense
    // location type (first of chain).
    // Example:
    // ```
    // field(edges) denseL;
    // field(edges, e->c->v) sparseR;
    // ...
    // expr = (denseL + sparseR);
    // ```
    // Result of expression (= iteration space) is sparse dimension: e->c->v
    else if(unstructuredDimLeft.getDenseLocationType() ==
            unstructuredDimRight.getDenseLocationType()) {
      // Propagate sparse
      curDimensions_ = right;

    } else { // No other subcase is allowed.
      dimensionsConsistent_ = false;
    }

  } else if(unstructuredDimLeft.isSparse() && unstructuredDimRight.isDense()) {

    // Case 2.a: Left is sparse, right is dense, right's location type matches left's target
    // location type (last of chain).
    // Example:
    // ```
    // field(edges, e->c->v) sparseL;
    // field(vertices) denseR;
    // ...
    // expr = (sparseL + denseR);
    // ```
    // Result of expression (= iteration space) is sparse dimension: e->c->v
    if(unstructuredDimLeft.getLastSparseLocationType() ==
       unstructuredDimRight.getDenseLocationType()) {
      // Propagate sparse
      curDimensions_ = left;
    }
    // Case 2.b: Left is sparse, right is dense, right's location type matches left's dense
    // location type (first of chain).
    // Example:
    // ```
    // field(edges, e->c->v) sparseL;
    // field(edges) denseR;
    // ...
    // expr = (sparseL + denseR);
    // ```
    // Result of expression (= iteration space) is sparse dimension: e->c->v
    else if(unstructuredDimLeft.getDenseLocationType() ==
            unstructuredDimRight.getDenseLocationType()) {
      // Propagate sparse
      curDimensions_ = left;

    } else { // No other subcase is allowed.
      dimensionsConsistent_ = false;
    }
  }
  // Case 3: Both operands are dense. They must match.
  // example:
  // ```
  // field(edges) denseL, denseR;
  // ...
  // expr = (denseL + denseR);
  // ```
  // Result of expression (= iteration space) is dense dimension edges
  else if(unstructuredDimLeft.isDense() && unstructuredDimRight.isDense()) {

    if(unstructuredDimLeft.getDenseLocationType() != unstructuredDimRight.getDenseLocationType()) {
      dimensionsConsistent_ = false;
      return;
    }
    curDimensions_ = left; // pick one, they are the same

  } else {
    dawn_unreachable("All cases should be covered.");
  }
}

void UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl::visit(
    const std::shared_ptr<iir::BinaryOperator>& binOp) {
  if(!dimensionsConsistent_) {
    return;
  }

  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl left(nameToDimensions_,
                                                                      idToNameMap_);
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl right(nameToDimensions_,
                                                                       idToNameMap_);

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
                                                                      idToNameMap_);
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl right(nameToDimensions_,
                                                                       idToNameMap_);

  assignmentExpr->getLeft()->accept(left);
  assignmentExpr->getRight()->accept(right);

  // type check failed further down below
  if(!(left.isConsistent() && right.isConsistent())) {
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
    const std::shared_ptr<iir::ReductionOverNeighborExpr>& reductionExpr) {
  if(!dimensionsConsistent_) {
    return;
  }

  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl init(nameToDimensions_,
                                                                      idToNameMap_);
  UnstructuredDimensionChecker::UnstructuredDimensionCheckerImpl ops(nameToDimensions_,
                                                                     idToNameMap_);

  reductionExpr->getInit()->accept(init);
  reductionExpr->getRhs()->accept(ops);

  // initial value needs to be consistent with operations on right hand side
  if(init.hasDimensions() && ops.hasDimensions()) {
    // As init and rhs get combined through a binary operation, let's reuse the same code
    checkBinaryOpUnstructured(init.getDimensions(), ops.getDimensions());
  }

  if(!dimensionsConsistent_) {
    return;
  }

  // if the rhs subtree has dimensions, we must check that such dimensions are consistent with the
  // declared rhs and lhs location types
  if(ops.hasDimensions()) {
    const auto& rhsUnstructuredDim = getUnstructuredDim(ops.getDimensions());
    dimensionsConsistent_ =
        rhsUnstructuredDim.isSparse()
            ? (rhsUnstructuredDim.getLastSparseLocationType() == reductionExpr->getRhsLocation() &&
               rhsUnstructuredDim.getDenseLocationType() == reductionExpr->getLhsLocation())
            : (rhsUnstructuredDim.getDenseLocationType() == reductionExpr->getRhsLocation());
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